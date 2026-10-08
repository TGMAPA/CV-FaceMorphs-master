# Yaml sharing

conda activate demo
conda env export --no-builds > environment.yml
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
## Link CUDA libraries for tf 
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh


# Identical Package Sharing
conda install -c conda-forge conda-pack
conda pack -n demo -o env_demo.tar.gz