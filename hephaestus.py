import faiss
from icecream import ic
import numpy as np
import jax
import jax.numpy as jnp
import optax
import logging
from threading import Lock


class Euclidean(object):
    @staticmethod
    def name():
        return "euclidean"

    @staticmethod
    @jax.jit
    def __call__(query, data):
        return jnp.linalg.norm(data - query, axis=1)

    @staticmethod
    def fixup_point(x):
        return x

    @staticmethod
    def fixup_gradient(grad, _x):
        return grad


class Angular(object):
    @staticmethod
    def name():
        return "angular"

    @staticmethod
    @jax.jit
    def __call__(query, data):
        return 1 - jnp.dot(data, query)

    @staticmethod
    def fixup_point(x):
        out = x / jnp.linalg.norm(x)
        assert x.shape == out.shape
        return out

    @staticmethod
    def fixup_gradient(grad, x):
        # project the gradients on the tangent plane
        grad = grad - jnp.dot(grad, x) * x
        return grad / jnp.linalg.norm(grad)


def relative_contrast(query, data, k, dist_fn):
    dists = dist_fn(query, data)
    idxs = jnp.argpartition(dists, k)
    kth = dists[idxs[k]]
    avg = jnp.mean(dists)
    return avg / kth


def local_intrinsic_dimensionality(query, data, k, dist_fn):
    dists = dist_fn(query, data)
    idxs = jnp.argpartition(dists, k)
    dists = dists[idxs[:k]]
    w = jnp.max(dists)
    half_w = 0.5 * w

    dists = dists[dists > 1e-5]

    small = dists[dists < half_w]
    large = dists[dists >= half_w]

    s = np.log(small / w).sum() + np.log1p((large - w) / w).sum()
    valid = small.size + large.size

    return -valid / s


def query_expansion(query, data, k, dist_fn):
    dists = dist_fn(query, data)
    dists = jnp.sort(dists)
    return ic(dists[2 * k]) / ic(dists[k])


def partition_by(candidates, fun):
    # first do an exponential search
    upper = 0
    lower = 0
    cur_res = None
    while upper < len(candidates):
        res = fun(candidates[upper])
        if res is not None:
            cur_res = res
            break
        lower = upper
        upper = upper * 2 if upper > 0 else 1
    upper = min(upper, len(candidates))

    # now we know that the predicate is satisfied between prev_ids (where it
    # is not satisfied) and cur_idx (where it is satisfied). So we do a binary search between the two
    while lower < upper:
        mid = (lower + upper) // 2
        mid_res = fun(candidates[mid])
        if mid_res is not None:
            cur_res = mid_res
            upper = mid
        else:
            lower = mid + 1

    return cur_res


def compute_recall(ground_distances, run_distances, count, epsilon=1e-3):
    """
    Compute the recall against the given ground truth, for `count`
    number of neighbors.
    """
    t = ground_distances[count - 1] + epsilon
    actual = 0
    for d in run_distances[:count]:
        if d <= t:
            actual += 1
    return float(actual) / float(count)


EMPIRICAL_HARDNESS_LOCK = Lock()


class IVFEmpiricalHardness(object):
    def __init__(self, distance_fn, recall):
        self.recall = recall
        self.distance_fn = distance_fn

    def fit(self, data):
        self.data = data
        self.nlists = int(jnp.sqrt(data.shape[1]))
        if self.distance_fn.name() == "euclidean":
            self.index = faiss.IndexIVFFlat(
                faiss.IndexFlatL2(data.shape[1]),
                data.shape[1],
                self.nlists,
                faiss.METRIC_L2,
            )
        else:
            # TODO: support the angular case
            raise ValueError("unsupported distance")
        self.query_params = list(range(1, self.nlists))
        self.index.train(data)
        self.index.add(data)

    def evaluate(self, query, k):
        ground_truth = jnp.sort(self.distance_fn(query, self.data))
        query = query.reshape(1, -1)

        def tester(nprobe):
            # we need to lock the execution because the statistics collection is
            # not thread safe, in that it uses global variables.
            with EMPIRICAL_HARDNESS_LOCK:
                faiss.cvar.indexIVF_stats.reset()
                self.index.nprobe = nprobe
                run_dists = jnp.sqrt(self.index.search(query, k)[0][0])
                distcomp = (
                    faiss.cvar.indexIVF_stats.ndis
                    + faiss.cvar.indexIVF_stats.nq * self.index.nlist
                )

            rec = compute_recall(ground_truth, run_dists, k)
            if rec >= self.recall:
                return distcomp / self.index.ntotal
            else:
                return None

        dist_frac = partition_by(self.query_params, tester)
        return dist_frac


class HNSWEmpiricalHardness(object):
    """
    Stores (and possibly caches on a file) a FAISS-HNSW index to evaluate the difficulty
    of queries, using the number of computed distances as a proxy for the difficulty.
    """

    def __init__(self, distance_fn, recall, index_params="HNSW32"):
        self.index_params = index_params
        self.recall = recall
        self.distance_fn = distance_fn

    def fit(self, data):
        # TODO: handle the angular distance case
        self.index = faiss.index_factory(data.shape[1], self.index_params)
        self.index.train(data)
        self.index.add(data)
        self.data = data

    def evaluate(self, query, k):
        """Evaluates the empirical difficulty of the given point `x` for the given `k`.
        Returns the number of distance computations, scaled by the number of datasets.
        """
        ground_truth = jnp.sort(self.distance_fn(query, self.data))
        query = query.reshape(1, -1)

        def tester(efsearch):
            # we need to lock the execution because the statistics collection is
            # not thread safe, in that it uses global variables.
            with EMPIRICAL_HARDNESS_LOCK:
                faiss.cvar.hnsw_stats.reset()
                self.index.hnsw.efSearch = efsearch
                run_dists = jnp.sqrt(self.index.search(query, k)[0][0])
                stats = faiss.cvar.hnsw_stats
                distcomp = stats.ndis

            rec = compute_recall(ground_truth, run_dists, k)
            if rec >= self.recall:
                return distcomp / self.index.ntotal
            else:
                return None

        dist_frac = partition_by(list(range(1, self.index.ntotal)), tester)

        if dist_frac is not None:
            return dist_frac
        else:
            raise Exception(
                "Could not get the desired recall, even visiting the entire dataset"
            )


class HephaestusGradient(object):
    def __init__(
        self,
        distance,
        scorer,
        learning_rate=1.0,
        max_iter=1000,
        seed=1234,
    ):
        self.distance = distance
        self.scorer = scorer
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = jax.random.key(seed)

    def fit(self, data):
        self.data = jnp.array(data)

    def generate(self, k, score_low, score_high):
        optimizer = optax.adam(self.learning_rate)
        grad_fn = jax.value_and_grad(relative_contrast)

        x = self.data[
            jax.random.randint(self.random_state, (1,), 0, self.data.shape[0])[0]
        ]
        x += jax.random.normal(self.random_state, (self.data.shape[1],)) * 0.001
        x = self.distance.fixup_point(x)
        opt_state = optimizer.init(x)

        for i in range(self.max_iter):
            score, grads = grad_fn(x, self.data, k, self.distance)
            logging.info("iteration %d score=%f", i, score)
            assert jnp.isfinite(score)

            if score_low <= score <= score_high:
                break

            grads = self.distance.fixup_gradient(grads, x)

            if score < score_low:
                # If we are too low, go the other way around
                grads = -grads

            updates, opt_state = optimizer.update(grads, opt_state)
            x = optax.apply_updates(x, updates)
            x = self.distance.fixup_point(x)

            assert jnp.all(jnp.isfinite(x))

        return x


if __name__ == "__main__":
    import h5py
    import matplotlib.pyplot as plt
    import faiss

    logging.basicConfig(level=logging.INFO)

    with h5py.File("fashion-mnist-784-euclidean.hdf5") as hfp:
        data = jnp.array(hfp["train"][:])
        d = data.shape[1]

    fig, axs = plt.subplots(1, 3, figsize=(6, 3))

    k = 10
    empirical_ivf = IVFEmpiricalHardness(Euclidean(), 0.9)
    empirical_ivf.fit(data)
    empirical_hnsw = HNSWEmpiricalHardness(Euclidean(), 0.9)
    empirical_hnsw.fit(data)

    distance = Euclidean()

    for target, ax in zip([4, 1.2, 1.05], axs):
        hg = HephaestusGradient(
            distance, relative_contrast, learning_rate=1, max_iter=500, seed=1234
        )
        hg.fit(data)
        q = hg.generate(k=k, score_low=0.99 * target, score_high=1.01 * target)
        rc = relative_contrast(q, data, k, distance)
        lid = local_intrinsic_dimensionality(q, data, k, distance)
        emp_ivf = empirical_ivf.evaluate(q, k) * 100
        emp_hnsw = empirical_hnsw.evaluate(q, k) * 100
        ic(emp_hnsw)
        ax.imshow(q.reshape(28, 28))
        ax.set_title(f"RC={rc:.2f}\nLID={lid:.2f}\nempirical_ivf={emp_ivf:.1f}\nempirical_hnsw={emp_hnsw:.1f}%")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("plot.png")
