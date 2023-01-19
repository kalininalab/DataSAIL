import shutil

from tests.data.generate import generate


def pytest_sessionstart(session):
    return
    generate(0.7, 10, "data/perf_7_3")
    generate(0.7, 100, "data/perf_70_30")


def pytest_sessionfinish(session):
    return
    shutil.rmtree("data/perf_7_3")
    shutil.rmtree("data/perf_70_30")
