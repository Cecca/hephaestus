# Hephaestus

Like the Greek god of artisans and blacksmiths forged the weapons of gods,
this software forges workloads for approximate nearest neighbor search algorithms.

## What does it do?

Consider the classic [fashion-mnist](https://github.com/zalandoresearch/fashion-mnist) dataset
(which can also be downloaded in HDF5 format from [ann-benchmarks](http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5)).
It contains 28x28 images of apparel items, that can be compared by the Euclidean distance.

Using `hephaestus.py`, we can create three queries with different levels of hardness
(more on this below):

![](imgs/queries-by-rc.png)

The top row of plots reports the synthesized queries, from left to right the grow harder to answer.
The tiny gray images below each query report the 10 nearest neighbors of each query. You can see that
harder queries have a more diversified set of answers.

## How is hardness measured?

We consider two different ways of measuring hardness: _explicative_ hardness measures try to capture
the geometric relationship between a query and the dataset, while _empirical_ hardness measures capture the
effort invested by an index data structure to answer a query.

For instance, the _Relative Constrast_ is an explicative difficulty measure defined as the ratio between the average 
distance of a query point with the dataset with the distance between the query and its $k$-th nearest neighbor.

$RC_k(q) = \frac{\sum_{x \in S} d(q, x) / n}{d(q, x_{(k)}}$

where $S$ is the dataset and $n=|S|$ is its size, and $x_{(k)}$ is the $k$-th nearest neighbor of $q$.
Queries with a smaller RC value are expected to be harder.

Empirical hardness measures are defined in terms of a specific index data structure (e.g. `faiss`'s IVF or HNSW) and a
_target recall_ value. In particular, the empirical hardness $\mathcal{H}_{\mathcal{D}, \rho}(q)$ for an index 
$\mathcal{D}$ at recall $\rho$ is the percentage of distance computations (wrt the number of points $n$) 
required by the index to answer the query $q$ with recall at least $\rho$.
Clearly, queries with a higher empirical hardness are harder to answer.

Going back to the figure above you can see that the easiest query has a hardness of about 4, and indeed the IVF index has an empirical hardness of about 4.6%, while HNSW computes only 0.3% of the total number of distances. Moving to the hardest query of the three (relative contrast 1.09) the indices have a much harder time: IVF computes 45.1% of the distances, HNSW 12.8%.

## How can I use it?

The dependencies are managed using [`uv`](https://docs.astral.sh/uv/). Hence the simplest way to run the script is via the command:

    uv run python hephaestus.py



