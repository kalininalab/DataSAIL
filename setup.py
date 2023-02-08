from setuptools import setup, find_packages


with open("README.md", "r") as desc_file:
    long_description = desc_file.read()


setup(
    name="DataSAIL",
    version="0.0.1",
    description="Data Splitting Against Information Leaking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='Apache Commons 2.0',
    author="Roman Joeres",
    maintainer="Roman Joeres",

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

    package_data={

    },
    python_requires=">=3.8, <4",
    keywords="bioinformatics",
    entry_points={
        "console_scripts": ["datasail = datasail.sail:sail"],
    },
)
