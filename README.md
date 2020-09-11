# SINVAD

This is a replication package for the ICSEW'20 paper [SINVAD: Search-based Image Space Navigation for DNN Image Classifier Test Input Generation](https://arxiv.org/abs/2005.09296).

## Overview
The `ipynb` files contain code for the experiments done in the SINVAD paper. 
They are presented in notebook form to facilitate interactive experimentation. 
The subdirectories contain code to train the models that are used in SINVAD experiments.

## Setup

Set up a python3 environment in your preferred way, then
```
pip install -r requirements.txt
```

For each research question's experiments, the prerequisite models are trained using the bash scripts in `init_scripts/`.
To run `RQx.ipynb`, you would first run
```
cd init_scripts
sh init_rqx.sh
```
where x is replaced with the desired RQ number.

Alternatively, to train models for all the RQs, one can run the `all_init.sh` script in the same directory.

## Run
Set up a Jupyter server and run the notebooks.
