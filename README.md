# Differentiable Stack

Impements Categorical Traffic Transformer in the environment of diffstack.

Paper [pdf](https://arxiv.org/abs/2311.18307)

## Setup 

Clone the repo with the desired branch. Use `--recurse-submodules` to also clone various submodules

For trajdata, we need to use branch `vectorize`, there are two options:

1. clone from NVlabs and then apply a patch

```
git clone --recurse-submodules --branch main git@github.com:NVlabs/trajdata.git;
cd trajdata;
git fetch origin
git reset --hard 748b8b1
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


# Fix opencv compatibility issue https://github.com/opencv/opencv-python/issues/591
pip uninstall opencv-python opencv-python-headless -y
pip install "opencv-python-headless==4.2.0.34"
# pip install "opencv-python-headless==4.7.0.72" # for python 3.9

# Sometimes you need to reinstall matplotlib with the correct version 

pip install matplotlib==3.3.4

```

### Key files and code structure

Diffstack uses a similar config system as [TBSIM](https://github.com/NVlabs/traffic-behavior-simulation), where the config templates are first defined in python inside the [diffstack/configs](/diffstack/configs/) folder. We separate the configs for [data](/diffstack/configs/trajdata_config.py), [training](/diffstack/configs/base.py), and [models](/diffstack/configs/algo_config.py).

The training and evaluation process takes in a JSON file as config, and one can call the [generate_config_templates.py](/diffstack/scripts/generate_config_templates.py) to generate all the template JSON configs, stored in [config/templates](/config/templates/) folder, by taking the default values from the python config files.

The models are separetely defined in the [models](/diffstack/models/) folder and [modules](/diffstack/modules/) folder where the former defines the model architecture, the latter wraps the torch model in a unified format called module, defined in [diffstack/modules/module.py](/diffstack/modules/module.py). 

Modules can be chained together to form [stacks](/diffstack/stacks/), which can be trained/evalulated as a whole. For this codebase, we only include CTT, thus the only type of stack is a prediction stack. 

A stack is wrapped as a Pytorch-lightning model for training and evaluation, see [train_pl.py](/diffstack/scripts/train_pl.py) for details.

The main files of CTT to look for is the [model file](/diffstack/models/CTT.py), and the [module file](/diffstack/modules/predictors/CTT.py).

We also included a rich collection of [utils functions](/diffstack/utils/), among which many are not used by CTT, but we believe they contribute to creating a convenient code base.

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

## Trained models



| Training dataset | Step time    | History horizon   | Future horizon | config | checkpoint |
|------------------|-------|------|----|--------|------------|
| nuScenes         | 0.25s | 1.5s | 3s | [config](https://drive.google.com/file/d/1fnPX0o2qPVGszFxbYX_LDSYJ221IUFk7/view?usp=drive_link)       | [ckpt](https://drive.google.com/file/d/1KvTdJQIEtk50cwiUzMFdl-ZtxpKg52kK/view?usp=drive_link)           |
| nuPlan           | 0.25s | 1.5s | 3s |[config](https://drive.google.com/file/d/1huNKKlTeT_i3oMOgPL6L1iUT2hKouKtt/view?usp=drive_link)        |[ckpt](https://drive.google.com/file/d/1w66sf6sTaoLI-Rl6MFHpuegu0y-i5R0y/view?usp=drive_link)            |
| WOMD             | 0.2s  | 1s   | 8s | [config](https://drive.google.com/file/d/1QgsHm3UhY74245YbhsQ4GlTyxyOBpE5y/view?usp=drive_link)       | [ckpt](https://drive.google.com/file/d/1qClV16V8jlSMMuPoAavFQeF7qJlb71GV/view?usp=drive_link)           |
