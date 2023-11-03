# SINVAD

This is a replication package for the TOSEM paper [Deceiving Humans and Machines Alike: Test Input Generation for DNNs].

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

For RQ4, you need to download the VAE models [from this link](doi.org/10.6084/m9.figshare.24482122) and place them in the `vae/models/` directory.

## Run
Set up a Jupyter server and run the notebooks.
