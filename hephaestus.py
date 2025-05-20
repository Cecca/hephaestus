from icecream import ic
import numpy as np
import jax
import jax.numpy as jnp
import optax


class Euclidean(object):
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
        ic(grad.shape, x.shape)
        grad = grad - ic(jnp.dot(grad, x)) * x
        return grad / jnp.linalg.norm(grad)


# @jax.jit
# def euclidean(data, query):
#     return jnp.linalg.norm(data - query, axis=1)


# @jax.jit
# def angular(data, query):
#     return 1 - jnp.dot(data, query)


# def k_smallest_dists(data, query, k, dist_fn):
#     dists = dist_fn(data, query)
#     idxs = jnp.argpartition(dists, k)
#     return jnp.sort(dists[idxs[:k]])


def relative_contrast(query, data, k, dist_fn):
    dists = dist_fn(query, data)
    ic(dists.shape)
    idxs = jnp.argpartition(dists, k)
    kth = dists[idxs[k]]
    avg = jnp.mean(dists)
    return avg / kth


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

    def generate(self, k, rc_low, rc_high):
        optimizer = optax.adam(self.learning_rate)
        grad_fn = jax.value_and_grad(relative_contrast)

        x = self.data[jax.random.randint(self.random_state, (1,), 0, self.data.shape[0])[0]]
        x += jax.random.normal(self.random_state, (self.data.shape[1], )) * 0.001
        x = self.distance.fixup_point(x)
        opt_state = optimizer.init(x)

        for i in range(self.max_iter):
            ic(i)
            rc, grads = grad_fn(x, self.data, k, self.distance)
            ic(rc, grads.shape)
            assert jnp.isfinite(rc)

            if rc_low <= rc <= rc_high:
                break

            grads = self.distance.fixup_gradient(grads, x)

            if rc < rc_low:
                grads = -grads
        
            updates, opt_state = optimizer.update(grads, opt_state)
            x = optax.apply_updates(x, updates)
            x = self.distance.fixup_point(x)

            assert jnp.all(jnp.isfinite(x))

        return x


if __name__ == "__main__":
    gen = np.random.default_rng(123)
    d = 100
    data = jnp.array(gen.normal(size=(100000, d)))
    data /= jnp.linalg.norm(data, axis=1)[:,jnp.newaxis]
    ic(data.shape)

    hg = HephaestusGradient(Euclidean(), relative_contrast, max_iter=100)
    hg.fit(data)
    q = hg.generate(k=10, rc_low=1.1, rc_high=1.2)

