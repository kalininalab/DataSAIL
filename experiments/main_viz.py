import sys
from pathlib import Path

from experiments.DTI.visualize import plot
from experiments.MPP.visualize import plot_double, heatmap_plot
from experiments.Strat.visualize import main


def plot_all(full_path: Path):
    plot(full_path / "DTI")
    plot_double(full_path.parent / "v03" / "MPP", ["QM8", "Tox21"])
    heatmap_plot(full_path.parent / "v03" / "MPP")
    main(full_path / "Strat")


if __name__ == '__main__':
    plot_all(Path(sys.argv[1]))
