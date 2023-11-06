#!/bin/bash

echo "Building conda package for Linux"
conda-build --variants "{'python': ['3.8', '3.9', '3.10', '3.11']}" -c conda-forge -c bioconda -c mosek --output-folder . --no-test recipe

echo "Convert conda package to other platforms"
conda convert --platform osx-64 linux-64/*.tar.bz2
conda convert --platform win-64 linux-64/*.tar.bz2
conda convert --platform osx-arm64 linux-64/*.tar.bz2

echo "Upload conda package to Anaconda Cloud"
anaconda upload --label main linux-64/*.tar.bz2
anaconda upload --label main osx-64/*.tar.bz2
anaconda upload --label main win-64/*.tar.bz2
anaconda upload --label main osx-arm64/*.tar.bz2
