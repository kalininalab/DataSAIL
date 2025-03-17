from setuptools import setup, find_packages
from datasail.version import __version__

with open("README.md", "r") as desc_file:
    long_description = desc_file.read()

setup(
    name="DataSAIL",
    version=__version__,
    description="Data Splitting Against Information Leaking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    author="Roman Joeres",
    maintainer="Roman Joeres",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    packages=["datasail"],
    include_package_data=True,
    # packages=find_packages(),
    # include_package_data=False,
    install_requires=[],
    python_requires=">=3.9, <4.0.0",
    keywords="bioinformatics",
)
