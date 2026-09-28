TOKEN=$1

for python in 3.9 3.10 3.11 3.12; do
    for version in lite full; do
        echo "=====" $version "===" $python "======================="
        # if [[ "$version" != "lite" && "$python" == "3.9" ]]; then
        #     echo "Skipping $version version with python $python"
        #     continue
        # fi
        conda create -n builder -y "python=$python"
        source /home/rjo21/miniconda3/bin/activate builder
        mamba install -y anaconda-client conda-build git
        if [ $version == "full" ]; then
            # echo "Build $version version with python $python"
            conda-build -q -c conda-forge -c bioconda --output-folder . --no-test --override-channels --package-format tar.bz2 --append-file base_recipe.yaml recipe
        else
            # echo "Build $version version with python $python"
            conda-build -q -c conda-forge -c bioconda --output-folder . --no-test --override-channels --package-format tar.bz2 --append-file base_recipe.yaml recipe_lite
            
            # echo "Convert to osx-arm64 $python lite"
            conda convert -p osx-arm64 linux-64/*.tar.bz2

            # echo "Convert to win64 $python lite"
            conda convert -p win-64 linux-64/*.tar.bz2

        fi
        # echo "Convert to osx-arm64 $python $version"
        conda convert -p osx-64 linux-64/*.tar.bz2

        source /home/rjo21/miniconda3/bin/deactivate
        conda remove -n builder --all -y
    done
done

anaconda -t $TOKEN upload linux-64/*.tar.bz2 --skip
anaconda -t $TOKEN upload osx-64/*.tar.bz2 --skip
anaconda -t $TOKEN upload osx-arm64/*.tar.bz2 --skip
anaconda -t $TOKEN upload win-64/*.tar.bz2 --skip
