import numpy as np

from scala.bqp.algos.cluster_cold_single import solve_ccx_iqp

items = []
sims = []
with open("tests/data/amay/pairwise_distance.tsv", "r") as data:
    for i, line in enumerate(data.readlines()[1:]):
        parts = line.strip().split("\t")
        items.append(parts[0])
        sims.append([float(x) for x in parts[1:]])

sims = np.array(sims)

split = solve_ccx_iqp(
    clusters=items,
    weights=[1 for _ in items],
    similarities=sims,
    threshold=0.0,
    limit=0.05,
    splits=[0.8, 0.2],
    names=["train", "test"],
    max_sec=1000,
    max_sol=1000,
)

print(split)
