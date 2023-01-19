import numpy as np
from sklearn.cluster import AgglomerativeClustering

from scala.bqp.algos.cluster_sim_cold_single import solve_ccx_iqp

file = "tests/data/amay/pairwise_distance.tsv"
items = []
dists = []
with open(file, "r") as data:
    for i, line in enumerate(data.readlines()[1:]):
        parts = line.strip().split("\t")
        items.append(parts[0])
        dists.append([float(x) for x in parts[1:]])

dists = np.array(dists)
ca = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage="average", distance_threshold=np.average(dists) * 0.9)
labels = ca.fit_predict(dists)

print(labels)
counts = np.asarray(np.unique(labels, return_counts=True)).T
print(counts)

cluster_dists, cluster_count = np.zeros((max(labels) + 1, max(labels) + 1)), np.zeros((max(labels) + 1, max(labels) + 1))
for i in range(len(items)):
    for j in range(i + 1, len(items)):
        if labels[i] != labels[j]:
            cluster_dists[labels[i], labels[j]] += dists[i, j]
            cluster_count[labels[i], labels[j]] += 1

            cluster_dists[labels[j], labels[i]] += dists[i, j]
            cluster_count[labels[j], labels[i]] += 1
cluster_dists /= np.max((cluster_count + np.eye(max(labels) + 1), np.ones_like(cluster_count)))
cluster_sims = 1 - cluster_dists / np.max(cluster_dists)

split = solve_ccx_iqp(
    clusters=list(range(max(labels) + 1)),
    weights=[c for _, c in counts],
    similarities=cluster_sims,
    threshold=1.0,
    limit=0.1,
    splits=[0.8, 0.2],
    names=["train", "test"],
    max_sec=1000,
    max_sol=1000,
)

print(split)

with open("amay.tsv", "w") as out:
    print("Query", "Split", sep="\t", file=out)
    for n, i in zip(items, labels):
        print(n, split[i], sep="\t", file=out)
