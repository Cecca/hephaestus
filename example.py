from hephaestus import *
import logging
import jax.numpy as jnp
import h5py
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO)

with h5py.File("fashion-mnist-784-euclidean.hdf5") as hfp:
    data = jnp.array(hfp["train"][:])
    d = data.shape[1]

fig, axs = plt.subplots(1, 3, figsize=(6, 3))

k = 10
empirical = IVFEmpiricalHardness(Euclidean(), 0.9)
empirical.fit(data)

distance = Euclidean()

for target, ax in zip([4, 1.2, 1.05], axs):
    hg = HephaestusGradient(
        distance, relative_contrast, learning_rate=1, max_iter=500, seed=1234
    )
    hg.fit(data)
    q = hg.generate(k=k, score_low=0.99 * target, score_high=1.01 * target)
    rc = relative_contrast(q, data, k, distance)
    lid = local_intrinsic_dimensionality(q, data, k, distance)
    emp = empirical.evaluate(q, k) * 100
    ax.imshow(q.reshape(28, 28))
    ax.set_title(f"RC={rc:.2f}\nLID={lid:.2f}\nempirical={emp:.1f}%")
    ax.axis("off")

plt.tight_layout()
plt.savefig("imgs/queries-by-rc.png")
