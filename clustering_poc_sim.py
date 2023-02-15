import numpy as np
from sklearn.cluster import AffinityPropagation

from datasail.solver.scalar.cluster_cold_single import solve_ccs_bqp

file = "tests/data/pipeline/prot_sim.tsv"
items = []
dists = []
with open(file, "r") as data:
    for i, line in enumerate(data.readlines()):
        parts = line.strip().split("\t")
        items.append(parts[0])
        dists.append([float(x) for x in parts[1:]])

dists = np.array(dists)
ca = AffinityPropagation(affinity='precomputed', random_state=42)
labels = ca.fit_predict(dists)

print(labels)
counts = np.asarray(np.unique(labels, return_counts=True)).T
print(counts)

cluster_sims, cluster_count = np.zeros((max(labels) + 1, max(labels) + 1)), np.zeros((max(labels) + 1, max(labels) + 1))
for i in range(len(items)):
    for j in range(i + 1, len(items)):
        if labels[i] != labels[j]:
            cluster_sims[labels[i], labels[j]] += dists[i, j]
            cluster_count[labels[i], labels[j]] += 1

            cluster_sims[labels[j], labels[i]] += dists[i, j]
            cluster_count[labels[j], labels[i]] += 1
cluster_sims /= (cluster_count + np.eye(max(labels) + 1))

print(cluster_sims)

print(np.average(cluster_sims))

split = solve_ccs_bqp(
    clusters=list(range(max(labels) + 1)),
    weights=[c for _, c in counts],
    similarities=cluster_sims,
    distances=None,
    threshold=np.average(cluster_sims),
    epsilon=0.1,
    splits=[0.7, 0.3],
    names=["train", "test"],
    max_sec=1000,
    max_sol=1000,
)

print(split)

splits = {}
for l in labels:
    if split[l] not in splits:
        splits[split[l]] = 0
    splits[split[l]] += 1
print(splits)
