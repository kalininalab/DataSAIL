from pathlib import Path

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


df = pd.read_csv("/home/rjo21/Downloads/LP_PDBBind.csv")["value"]
df.hist(bins=101)
plt.show()
exit(0)


train = pd.read_csv(Path("random_ref") / "train.csv")["Target"].values.tolist()
test = pd.read_csv(Path("random_ref") / "test.csv")["Target"].values.tolist()
combined = np.array(train + test)

vc = list(sorted(np.array(np.unique(combined, return_counts=True)).T.tolist(), key=lambda x: int(x[1]), reverse=True))
print(vc)
plt.plot(list(range(len(vc))), [int(x[1]) for x in vc])
plt.yscale("log")
plt.xscale("log")
plt.show()
# plt.hist(labels, bins=101)
# plt.show()
