import hephaestus
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.animation as animation
from icecream import ic


def simulate(
    n=30,
    k=1,
    outpath="example.png",
    seed=1234,
    with_legend=False,
    with_generate=False,
    target=1.4,
    delta=0.1,
    start=None,
    annotate=None,
):
    assert k > 0
    gen = np.random.default_rng(seed)
    xs = gen.uniform(0, 10, n)
    ys = gen.uniform(0, 10, n)

    points = np.vstack((xs, ys)).T

    padding = 1
    min_x, max_x = np.min(points[:, 0]) - padding, np.max(points[:, 0]) + padding
    min_y, max_y = np.min(points[:, 1]) - padding, np.max(points[:, 1]) + padding
    num_samples = 500

    grid_x = np.linspace(min_x, max_x, num_samples)
    grid_y = np.linspace(min_y, max_y, num_samples)

    rcs = np.zeros((num_samples, num_samples))

    for i, x in enumerate(grid_x):
        for j, y in enumerate(grid_y):
            q = np.array([x, y])
            dists = np.linalg.norm(points - q, axis=1)
            dists = np.sort(dists)
            nearest = dists[k - 1]  # .min()
            rc = dists.mean() / nearest
            assert rc > 1
            rcs[i, j] = rc

    fig = plt.figure(figsize=(5, 5))
    plt.imshow(
        rcs.T, norm=LogNorm(), origin="lower", extent=(min_x, max_x, min_y, max_y)
    )
    plt.scatter(xs, ys, c="white", edgecolor="black", linewidth=2, s=20, zorder=100)
    plt.gca().set_aspect("equal")

    if with_legend:
        plt.colorbar(
            ax=plt.gca(),
            norm=LogNorm(),
            values=np.linspace(rcs.min(), rcs.max(), 100),
            # ticks = [1, 100, 200, 300, 400, 500, 600]
        )
    if with_generate:
        if start is None:
            start = np.array(
                [
                    min_x + (max_x - min_x) / 2,
                    min_y + (max_y - min_y) / 2,
                ]
            )
            ic(start)

        generator = hephaestus.HephaestusGradient(
            hephaestus.Euclidean(), hephaestus.relative_contrast, trace=True
        )
        generator.fit(points)
        q = generator.generate(k, score_low=target - delta, score_high=target + delta, start=start)
        path = np.array(generator.trace)
        ic(path)

        dists = np.linalg.norm(points - q, axis=1)
        dists = np.sort(dists)
        nearest = dists[k - 1]
        rc = dists.mean() / nearest
        ic(rc)
        # line = plt.plot(path[:, 0], path[:, 1], c="white")
        # pts = plt.scatter(q[0], q[1], marker="*", c="white", ec="black", s=220, zorder=100)
        line = plt.plot([path[0, 0]], [path[0, 1]], c="white")[0]
        pts = plt.scatter([], [], marker="*", c="white", ec="black", s=220, zorder=100)
        def update(frame):
            line.set_xdata(path[:frame+1,0])
            line.set_ydata(path[:frame+1,1])
            if frame == path.shape[0]-1:
                pts.set_offsets(q)
            return line, pts

        ani = animation.FuncAnimation(fig=fig, func=update, frames=path.shape[0], interval=1000)
    else:
        ani = None

    if annotate is not None:
        dists = np.linalg.norm(points - np.array(annotate), axis=1)
        dists = np.sort(dists)
        nearest = dists[k - 1]  # .min()
        rc = dists.mean() / nearest
        ic(rc)
        if rc > 2:
            color = "white"
        else:
            color = "black"
        plt.scatter(annotate[0], annotate[1], c=color, marker="x", s=100)

    plt.axis("off")
    plt.tight_layout(pad=0)
    if ani is not None:
        ani.save(filename=outpath, writer="pillow")
    else:
        plt.savefig(outpath)


marker = (3.47, 2.72)
simulate(
    k=10,
    outpath="imgs/example-k10.png",
    with_generate=False,
    annotate=marker,
)
simulate(
    k=10,
    outpath="imgs/example-k10-with-path.gif",
    with_generate=True,
    target=1.4,
    delta=0.01,
)
