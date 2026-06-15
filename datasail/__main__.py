import os
from pathlib import Path

if "GRB_LICENSE_FILE" not in os.environ or not Path(os.environ["GRB_LICENSE_FILE"]).exists():
    os.environ["GRB_LICENSE_FILE"] = "/home/rjo21/gurobi_mickey.lic"

from datasail.sail import sail

if __name__ == '__main__':
    sail()
