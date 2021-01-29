# iFlow

This repository is about reproducing and extending the results presented in the paper "Identifying through Flows for Recovering Latent Representations" by Shen Li et al. and is forked from the original repository which can be found [here](https://github.com/MathsXDC/iFlow).
This is part of the Fairness, Accountability, Confidentiality and Transparency in AI course at the University of Amsterdam. 

The iFlow model aims to recover the true latent variables through the concept of identifiability.

## Dependencies

### Installation

#### Linux

```
conda env create --name envname --file=environment_linux.yml
```

#### Windows

```
conda env create --name envname --file=environment_windows.yml
```


### Instructions

To run the experiments run Jupyter Notebook 

```jupyter notebook```

and following the instructions in `results.ipynb`.

To train a model run

```
x: argument string to generate a dataset. Usage explained in lib.data.create_if_not_exist_dataset.
i: Model type
ft: Flow type
npa: Natural parameter activation function
fl: Flow length
lr_df: Learning rate drop factor
lr_pn: Learning rate patience
b: Batch size
e: Epochs
l: Learning rate
s: Model seed
u: GPU ID
Add '-c' to use cuda GPU
Add '-p' to preload data on GPU for increased performance

python main_save_mcc.py 
        -x 1000_40_5_5_3_1_gauss_xtanh_u_f \
        -i iFlow \
        -ft RQNSF_AG \
        -npa Softplus \
        -fl 10 \
        -lr_df 0.25 \
        -lr_pn 10 \
        -b 64 \
        -e 20 \
        -l 1e-3 \
        -s 1 \
        -u 0 \
        -c \
        -p
```


### Authors

Max van den Heuvel, Roel Klein, Tim Stolp, Fengyuan Sun