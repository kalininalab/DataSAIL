PYTHON_V=$1
VERSION=$2
TOKEN=$3

conda create -n builder -y python=$PYTHON_V
source /home/rjo21/miniconda3/bin/activate builder
conda install anaconda-client conda-build git
if [ $VERSION == "full" ]; then
    conda-build -q -c conda-forge -c bioconda --output-folder . --no-test --override-channels --package-format tar.bz2 --append-file base_recipe.yaml recipe
else
    conda-build -q -c conda-forge -c bioconda --output-folder . --no-test --override-channels --package-format tar.bz2 --append-file base_recipe.yaml recipe_lite
    conda convert -p osx-arm64 linux-64/*.tar.bz2
    conda convert -p win-64 linux-64/*.tar.bz2
    anaconda -t $TOKEN upload osx-arm64/*.tar.bz2
    anaconda -t $TOKEN upload win-64/*.tar.bz2
fi
conda convert -p osx-64 linux-64/*.tar.bz2
anaconda -t $TOKEN upload linux-64/*.tar.bz2
anaconda -t $TOKEN upload osx-64/*.tar.bz2

source /home/rjo21/miniconda3/bin/deactivate
# conda remove -n builder --all -y