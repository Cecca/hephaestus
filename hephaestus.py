import math
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

    @staticmethod
    def from_euclidean(dists):
        return dists

    @staticmethod
    def random_neighbor(x, random_state, **kwargs):
        """Returns a random neighbor of point x"""
        scale = kwargs["scale"]
        direction = jax.random.normal(random_state, x.shape)
        direction /= jnp.linalg.norm(direction)
        amount = scale*jax.random.exponential(random_state)
        offset = direction * amount
        neighbor = x + offset
        return neighbor.astype(jnp.float32)


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

    @staticmethod
    def from_euclidean(dists):
        return 1 - (2 - dists**2) / 2

    @staticmethod
    def random_neighbor(x, random_state, **kwargs):
        """Returns a random neighbor of point x"""
        scale = kwargs["scale"]
        a = scale / (1-scale)
        b = 1

        # get orthogonal vector in a random direction
        y = jax.random.normal(random_state, shape=x.shape)
        y[0] = (0.0 - jnp.sum(x[1:] * y[1:])) / x[0]
        y /= jnp.linalg.norm(y)
        # get a random value for the dot product
        d = 1 - jax.random.beta(random_state, a, b)
        # compute the neighbor position
        neighbor = x + y * jnp.tan(np.arccos(d))
        neighbor /= jnp.linalg.norm(neighbor)
        # check that the dot product is what we expect
        d_check = jnp.dot(x, neighbor)
        assert jnp.isclose(d, d_check)
        return neighbor


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
    return dists[2 * k] / dists[k]


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
        self.assert_normalized = self.distance_fn.name() == "angular"

    def fit(self, data):
        if self.assert_normalized:
            assert jnp.allclose(1.0, jnp.linalg.norm(data, axis=1)), (
                "Data points should have unit norm"
            )
        self.data = data
        self.nlists = int(jnp.sqrt(data.shape[1]))
        self.index = faiss.IndexIVFFlat(
            faiss.IndexFlatL2(data.shape[1]),
            data.shape[1],
            self.nlists,
            faiss.METRIC_L2,
        )
        self.query_params = list(range(1, self.nlists))
        self.index.train(data)
        self.index.add(data)

    def evaluate(self, query, k):
        ground_truth = jnp.sort(self.distance_fn(query, self.data))
        query = query.reshape(1, -1)
        if self.assert_normalized:
            assert jnp.allclose(1.0, jnp.linalg.norm(query, axis=1)), (
                "Data points should have unit norm"
            )

        def tester(nprobe):
            # we need to lock the execution because the statistics collection is
            # not thread safe, in that it uses global variables.
            with EMPIRICAL_HARDNESS_LOCK:
                faiss.cvar.indexIVF_stats.reset()
                self.index.nprobe = nprobe
                run_dists = self.distance_fn.from_euclidean(
                    jnp.sqrt(self.index.search(query, k)[0][0])
                )
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
        self.assert_normalized = self.distance_fn.name() == "angular"

    def fit(self, data):
        if self.assert_normalized:
            assert jnp.allclose(1.0, jnp.linalg.norm(data, axis=1)), (
                "Data points should have unit norm"
            )
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
        if self.assert_normalized:
            assert jnp.allclose(1.0, jnp.linalg.norm(query, axis=1)), (
                "Data points should have unit norm"
            )

        def tester(efsearch):
            # we need to lock the execution because the statistics collection is
            # not thread safe, in that it uses global variables.
            with EMPIRICAL_HARDNESS_LOCK:
                faiss.cvar.hnsw_stats.reset()
                self.index.hnsw.efSearch = efsearch
                run_dists = self.distance_fn.from_euclidean(
                    jnp.sqrt(self.index.search(query, k)[0][0])
                )
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


class Generator(object):
    def next_rand_state(self):
        key, subkey = jax.random.split(self.random_state)
        self.random_state = key
        return subkey

    def fit(self, data):
        self.data = jnp.array(data)

    def generate(self, k, score_low, score_high):
        pass

    def generate_many(self, k, scores):
        from joblib import Parallel, delayed

        def fn(i, score_pair):
            for _ in range(i):
                # make sure different iterations use a different random seed
                self.next_rand_state()
            q = self.generate(k, score_pair[0], score_pair[1])
            rc = relative_contrast(q, self.data, k, self.distance)
            lid = local_intrinsic_dimensionality(q, self.data, k, self.distance)
            exp = query_expansion(q, self.data, k, self.distance)
            dists = self.distance(q, self.data)
            idxs = jnp.argsort(dists)
            return q, rc, lid, exp, idxs[:k], dists[idxs[:k]]

        res = Parallel(n_jobs=-1)(delayed(fn)(i, score_pair) for i, score_pair in enumerate(scores))
        queries, rcs, lids, exps, idxs, dists = zip(*res)
        return {
            "test": jnp.stack(queries),
            "relative_contrast": jnp.array(rcs),
            "local_intrinsic_dimensionality": jnp.array(lids),
            "query_expansion": jnp.array(exps),
            "neighbors": jnp.stack(idxs),
            "distances": jnp.stack(dists)
        }
    

class HephaestusAnnealing(Generator):
    def __init__(
        self,
        distance,
        scorer,
        initial_temperature=1.0,
        max_iter=1000,
        scale=0.1,
        seed=1234,
    ):
        self.distance = distance
        self.scorer = scorer
        self.initial_temperature = initial_temperature
        self.max_iter = max_iter
        self.scale = scale
        self.random_state = jax.random.key(seed)

    def temperature(self, step):
        # fast annealing schedule
        return self.initial_temperature / (step + 1)

    def generate(self, k, score_low, score_high):
        x = self.data[
            jax.random.randint(self.next_rand_state(), (1,), 0, self.data.shape[0])[0]
        ]
        x += jax.random.normal(self.next_rand_state(), (self.data.shape[1],)) * 0.001
        x = self.distance.fixup_point(x)
        y = self.scorer(x, self.data, k, self.distance)
        x_best, y_best = x, y
        logging.info("start from score %f", y)
        if score_low <= y <= score_high:
            logging.info("returning immediately")
            return x

        steps_since_last_improvement = 0
        steps_threshold = max(self.max_iter // 100, 10)
        logging.info("steps threshold %d", steps_threshold)

        for step in range(self.max_iter):
            if steps_since_last_improvement >= steps_threshold:
                logging.info("moving back to the previous best due to lack of improvement")
                x, y = x_best, y_best
                steps_since_last_improvement = 0

            x_next = self.distance.random_neighbor(x, self.next_rand_state(), scale=self.scale)
            y_next = self.scorer(x_next, self.data, k, self.distance)
            # FIXME: handle case of navigating towards easier points
            if score_low <= y_next <= score_high:
                logging.info(
                    "Returning query point with score %f after %d iterations",
                    y_next,
                    step,
                )
                return x_next
            elif min(abs(y_next - score_low), abs(y_next - score_high)) <= min(
                abs(y - score_low), abs(y - score_high)
            ):
                # the next candidate goes towards the desired range
                x, y = x_next, y_next
                logging.info(
                    "new best score %f, (still %f to go)",
                    y,
                    min(abs(y_next - score_low), abs(y_next - score_high)),
                )
                x_best, y_best = x, y
                steps_since_last_improvement = 0
            else:
                # we pick the neighbor by the Metropolis criterion
                delta = abs(y - y_next)
                t = self.temperature(step)
                p = math.exp(-delta / t)
                if jax.random.bernoulli(self.next_rand_state(), p):
                    x, y = x_next, y_next
                    logging.info(
                        "new score %f temperature %f (%d since last improvement, p=%f, delta=%f)",
                        y,
                        t,
                        steps_since_last_improvement,
                        p,
                        delta,
                    )
                steps_since_last_improvement += 1
            if step % 50 == 0:
                logging.info(
                    "%d/%d current score %f, (still %f to target)",
                    step,
                    self.max_iter,
                    y,
                    min(abs(y - score_low), abs(y - score_high)),
                )

        return x



class HephaestusGradient(Generator):
    def __init__(
        self,
        distance,
        scorer,
        learning_rate=1.0,
        max_iter=1000,
        seed=1234,
        trace=False,
    ):
        self.distance = distance
        self.scorer = scorer
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = jax.random.key(seed)
        if trace:
            self.trace = []
        else:
            self.trace = None

    def generate(self, k, score_low, score_high, start=None):
        optimizer = optax.adam(self.learning_rate)
        grad_fn = jax.value_and_grad(self.scorer)

        if start is None:
            x = self.data[
                jax.random.randint(self.next_rand_state(), (1,), 0, self.data.shape[0])[0]
            ]
            x += jax.random.normal(self.next_rand_state(), (self.data.shape[1],)) * 0.001
            x = self.distance.fixup_point(x)
        else:
            x = self.distance.fixup_point(start)
        opt_state = optimizer.init(x)

        for i in range(self.max_iter):
            if self.trace is not None:
                self.trace.append(x)
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


def main():
    import h5py
    import argparse
    import pathlib

    parser = argparse.ArgumentParser("hephaestus")
    parser.add_argument("-d", "--dataset", type=pathlib.Path, required=True)
    parser.add_argument("-k", type=int, required=True)
    parser.add_argument("--distance", type=str, choices=["euclidean", "angular"], required=False)
    parser.add_argument("--method", type=str, choices=["gradient", "annealing"], required=False, default="gradient")
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("-o", "--output", type=pathlib.Path, required=False)
    parser.add_argument("--verbose", "-v", action="count", default=0)
    parser.add_argument("-q", "--queries", action="extend", nargs="+", type=str)
    parser.add_argument("--learning-rate", type=float, default=1)
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--initial-temperature", type=float, default=1)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)

    args = parser.parse_args()
    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO)
    elif args.verbose >= 2:
        logging.basicConfig(level=logging.DEBUG)

    k = args.k

    path = args.dataset
    if not path.is_file():
        raise ValueError("dataset file does not exist!")
    if args.output is None:
        output = path.with_name(path.stem + "-queries" + path.suffix)
    else:
        output = args.output
    if output.is_file():
        raise ValueError("output file already exists!")

    if args.distance is None:
        if path.match("*euclidean*"):
            distance = Euclidean()
        elif path.match("*angular*"):
            distance = Angular()
        else:
            raise ValueError("distance not given, and cannot be inferred from name")
        logging.warning(
            "distance not given, inferred from file name: %s", distance.name()
        )
    elif args.distance == "euclidean":
        distance = Euclidean()
    elif args.distance == "angular":
        distance = Angular()
    else:
        raise ValueError("unsupported distance function: " + args.distance)

    scores = []
    delta = args.delta
    for spec in args.queries:
        n, rc = spec.split(":")
        n = int(n)
        rc = float(rc)
        scores.extend([(rc / (1 + delta), rc * (1 + delta))] * n)

    if args.method == "gradient":
        logging.info("Using the gradient method")
        generator = HephaestusGradient(
            distance, relative_contrast, args.learning_rate, args.max_iter, args.seed
        )
    elif args.method == "annealing":
        logging.info("Using the annealing method")
        generator = HephaestusAnnealing(
            distance, relative_contrast, initial_temperature=args.initial_temperature, max_iter=args.max_iter, scale=args.scale, seed=args.seed
        )
    else:
        raise ValueError(f"unsupported method `{args.method}`")

    with h5py.File(path) as hfp:
        data = jnp.array(hfp["train"][:])
    generator.fit(data)

    queries = generator.generate_many(k, scores)

    with h5py.File(output, "w") as hfp:
        metadata = vars(args)
        del metadata["queries"]
        for key, value in metadata.items():
            try:
                hfp.attrs[key] = value
            except TypeError:
                hfp.attrs[key] = str(value)
        for key, data in queries.items():
            ic(key, data)
            hfp[key] = data


if __name__ == "__main__":
    main()
