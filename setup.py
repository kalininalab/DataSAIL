from setuptools import setup, find_packages


with open("README.md", "r") as desc_file:
    long_description = desc_file.read()

with open("meta.yaml") as meta_file:
    version = meta_file.readlines()[2].split("\"")[1]

setup(
    name="DataSAIL",
    version=version,
    description="Data Splitting Against Information Leaking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='Apache Commons 2.0',
    author="Roman Joeres",
    maintainer="Roman Joeres",

    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    packages=find_packages(),
    setup_requires=['setuptools_scm'],
    include_package_data=True,
    install_requires=[

    ],

    package_data={

    },
    python_requires=">=3.10, <4.0.0",
    keywords="bioinformatics",
    entry_points={
        "console_scripts": ["datasail = datasail.sail:sail"],
    },
)
