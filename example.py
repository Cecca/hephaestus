from hephaestus import *
import logging
import jax.numpy as jnp
import h5py
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


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

# fig, axs = plt.subplots(2, len(targets), figsize=(6, 3))
fig = plt.figure(figsize=(6, 5), layout="constrained")
gs = GridSpec(
    k // 2 + 2, 2 * len(targets), figure=fig, height_ratios=[1.5] * 2 + [1] * 5
)

# generate three queries with different target Relative Constrast
# values. The first is expected to be easier than the last
for i, target in enumerate(targets):
    # set up the generator
    hg = HephaestusGradient(
        # distance metric to use
        distance,
        # the function used to score candidate queries
        relative_contrast,
        learning_rate=1,
        # how many iterations to run at most. Returns the best
        # candidate query found so far, if the budget is exhausted
        max_iter=500,
        # the seed makes the execution deterministic
        seed=1234,
    )
    # pass the data to the generator
    hg.fit(data)
    # generate a query whose relative contrast is within 1% of the target one
    q = hg.generate(k=k, score_low=0.99 * target, score_high=1.01 * target)

    # measure difficulties
    rc = relative_contrast(q, data, k, distance)
    lid = local_intrinsic_dimensionality(q, data, k, distance)
    emp_ivf = empirical_ivf(q, k) * 100
    emp_hnsw = empirical_hnsw(q, k) * 100

    # get the kth nearest neighbor
    neighbors_idxs = jnp.argpartition(distance(q, data), k)[:k]

    # plot the query
    ax = fig.add_subplot(gs[0:2, 2 * i : 2 * i + 2])
    ax.imshow(q.reshape(28, 28))
    ax.set_title(f"RC={rc:.2f}\nLID={lid:.2f}\nempirical_ivf={emp_ivf:.1f}%\nempirical_hnsw={emp_hnsw:.1f}%")
    ax.axis("off")

    # plot the neighbors
    for j, idx in enumerate(neighbors_idxs):
        ax = fig.add_subplot(gs[2 + (j % (k // 2)), 2 * i + (j // (k // 2))])
        ax.imshow(data[idx, :].reshape(28, 28), cmap="Greys")
        ax.axis("off")

plt.tight_layout()
plt.savefig("imgs/queries-by-rc.png")
