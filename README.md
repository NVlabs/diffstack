# Differentiable Stack

Impements Categorical Traffic Transformer in the environment of diffstack.

Paper [pdf](link)

## Setup 

Clone the repo with the desired branch. Use `--recurse-submodules` to also clone various submodules

For trajdata, we need to use branch `vectorize`, there are two options:

1. clone from NVlabs and then apply a patch

```
git clone --recurse-submodules --branch main git@github.com:NVlabs/trajdata.git;
cd trajdata;
git apply ../patches/trajdata_vectorize.patch
cd ..
```

2. clone from a forked repo of trajdata


```
git clone --recurse-submodules --branch vectorize git@github.com:chenyx09/trajdata.git
```

Then add Pplan

```
git clone --recurse-submodules git@github.com:NVlabs/spline-planner.git
```

You can also sync submodules later using
```
git submodule update --remote
```

### Install diffstack

We will install diffstack with a conda env.

Create a `conda` environment for `diffstack`:

```
conda create -n diffstack python=3.9
conda activate diffstack
```

Next install torch pytorch compatible to your CUDA setup following [Pytorch website](https://pytorch.org/get-started/locally/)



Install the required python packages for diffstack

```
pip install -r requirements.txt
```

Install submodules manually (use `-e` for developer mode)
```
pip install -e ./trajdata
pip install -e ./spline-planner
```



These additional steps might be necessary
```
# need to reinstall pathos, gets replaced by multiprocessing install 
pip uninstall pathos -y
pip install pathos==0.2.9

# Gpu affinity for cpu-gpu assignments on NGC (optional)
pip install git+https://gitlab-master.nvidia.com/dl/gwe/gpu_affinity

# On Mac sometimes we need to reinstall torch
conda install pytorch torchvision torchaudio -c pytorch

# The default requirements installs jax for cpu only. To enable jax with GPU, see https://github.com/google/jax#installation

# networkx package is not well aligned between py3.8 and py3.9, if you encounter an error for unknown module of gcd in fraction, manually modify that line of code for networkx. It should be located in the site-package/networkx/algorithms/dag.py:23
# from fractions import gcd
from math import gcd

# The version of numpy might be messed by merging diffstack + mm3d, try restore numpy version for mm3d if the test script does not work after install diffstack requirements
pip uninstall numpy
pip install numpy==1.23.5

# The version of bokeh might be messed if you see errors for 'module not found' when using nuplan, update bokeh version to bokeh==2.4.3
pip install bokeh==2.4.3

pip uninstall pygeos

# To parse CARLA OpenDrive maps manually install extra trajdata dependencties (can be removed once trajdata is updated)
pip install intervaltree bokeh==2.4.3 geopandas selenium
pip install -e ./trajdata/src/trajdata/dataset_specific/opendrive/custom_imap

# Fix opencv compatibility issue https://github.com/opencv/opencv-python/issues/591
pip uninstall opencv-python opencv-python-headless -y
pip install "opencv-python-headless==4.2.0.34"
# pip install "opencv-python-headless==4.7.0.72" # for python 3.9

# Sometimes you need to reinstall matplotlib with the correct version 

pip install matplotlib==3.3.4

```


### Data

CTT uses [trajdata](https://github.com/NVlabs/trajdata) as the dataloader, technically, you can train with any dataset supported by trajdata. Considering the vectorized map support, we have tested CTT with WOMD, nuScenes, and nuPlan.


## Generating config templates

```
python diffstack/scripts/generate_config_templates.py
```

## Training and eval

The following examples use nuScenes trainval as dataset, you'll need to prepare the nuScenes dataset following instructions in [trajdata](https://github.com/NVlabs/trajdata).

Training script:

```
python diffstack/scripts/train_pl.py 
--config_file=<path to CTT>/config/templates/CTTPredStack.json 
--remove_exp_dir
--dataset_path=<your dataset path>
```

Eval script:

```
python diffstack/scripts/train_pl.py
--evaluate
--config_file=<your config file>
--ckpt_path=<your checkpoint path>
--test_data=<data split for evaluation, e.g. val>
--test_data_root=<eval data root, e.g. nusc_trainval>
--log_image_frequency=10
--eval_output_dir=<a directory to save evaluation results>
--test_batch_size=16
--dataset_path=<your dataset path>
```

Training and eval example commands are also included in the `.vscode/launch.json` file.


