from setuptools import setup, find_packages


with open("README.md", "r") as desc_file:
    long_description = desc_file.read()

path_to_version_file = "./misc/_version.py"

with open(path_to_version_file) as version_file:
    exec(version_file.read().strip())

setup(
    name="SCALA",
    version="0.0.1",
    description="Sequence Clustering Against Leaking informAtion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='LGPL-2.1',
    author="Roman Joeres, Anne Tolkmitt, Alexander Gress",
    maintainer="Roman Joeres, Anne Tolkmitt, Alexander Gress",

    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    packages=find_packages(),
    setup_requires=['setuptools_scm'],
    include_package_data=True,
    install_requires=[
        "biopython>=1.78",
        "matplotlib>=3.3.2",
        "numpy>=1.22.3",
        "psutil>=5.8.0",
        "pandas>=1.3.3",
        "autopep8>=1.5.7",
    ],

    package_data = {

    },
    python_requires=">=3.8, <4",
    keywords="bioinformatics",
    entry_points={
        "console_scripts": ["scala = scala.scala_main:main"],
    },
)
