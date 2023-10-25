import os
import shutil

from tests.data.generate import generate


def pytest_sessionstart(session):
    generate(0.7, 10, "data/perf_7_3")
    generate(0.7, 100, "data/perf_70_30")


def pytest_sessionfinish(session):
    try:
        shutil.rmtree("data/perf_7_3", ignore_errors=True)
        shutil.rmtree("data/perf_70_30", ignore_errors=True)
        shutil.rmtree("data/pipeline/out/", ignore_errors=True)
        if os.path.isfile("data/solver.log"):
            os.remove("data/solver.log")
        if os.path.isfile("solver.log"):
            os.remove("solver.log")
    except OSError:
        pass
