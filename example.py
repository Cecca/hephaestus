from hephaestus import *
import logging
import jax.numpy as jnp
import h5py
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO)

with h5py.File("fashion-mnist-784-euclidean.hdf5") as hfp:
    data = jnp.array(hfp["train"][:])
    d = data.shape[1]


k = 10
targets = [4, 1.2, 1.05]

# the distance function to use
distance = Euclidean()

# empirical difficulty hardness measures
empirical_ivf = IVFEmpiricalHardness(distance, 0.9)
empirical_ivf.fit(data)
empirical_hnsw = HNSWEmpiricalHardness(distance, 0.9)
empirical_hnsw.fit(data)

fig, axs = plt.subplots(2, len(targets), figsize=(6, 3))

# generate three queries with different target Relative Constrast
# values. The first is expected to be easier than the last
for i, target in enumerate(targets):
    # set up the generator
    hg = HephaestusGradient(
        distance, relative_contrast, learning_rate=1, max_iter=500, seed=1234
    )
    # pass the data to the generator
    hg.fit(data)
    # generate a query whose relative contrast is within 1% of the target one
    q = hg.generate(k=k, score_low=0.99 * target, score_high=1.01 * target)

    # measure difficulties
    rc = relative_contrast(q, data, k, distance)
    lid = local_intrinsic_dimensionality(q, data, k, distance)
    emp_ivf = empirical_ivf.evaluate(q, k) * 100
    emp_hnsw = empirical_hnsw.evaluate(q, k) * 100

    # get the kth nearest neighbor
    kth_index = jnp.argpartition(distance(q, data), k)[k]
    neighbor = data[kth_index]

    # plot the query
    axs[0, i].imshow(q.reshape(28, 28))
    axs[0, i].set_title(f"RC={rc:.2f}\nLID={lid:.2f}\nempirical={emp_ivf:.1f}%")
    axs[0, i].axis("off")

    # plot the neighbor
    axs[1, i].imshow(neighbor.reshape(28, 28))
    axs[1, i].set_title("k-th neighbor")
    axs[1, i].axis("off")

plt.tight_layout()
plt.savefig("imgs/queries-by-rc.png")
