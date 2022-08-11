from setuptools import setup, find_packages

path_to_readme = "ReadMe.txt"

if path_to_readme is not None:
    with open(path_to_readme, "r") as desc_file:
        long_description = desc_file.read()
else:
    long_description = ''

path_to_version_file = "./misc/_version.py"

with open(path_to_version_file) as version_file:
    exec(version_file.read().strip())

setup(
    name="SCALA",
    version=__version__,
    description="Sequence Clustering Against Leaking informAtion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='LGPL-2.1',
    author="Anne Tolkmitt, Alexander Gress",
    maintainer="Anne Tolkmitt, Alexander Gress",

    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    packages=find_packages(),
    setup_requires=['setuptools_scm'],
    include_package_data=True,
    install_requires=[
        "biopython>=1.79",
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
