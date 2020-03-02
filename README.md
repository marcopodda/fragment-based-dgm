# Code for the paper "A Deep Generative Model for Fragment-Based Molecule Generation" (AISTATS 2020)

### Installation

Run:

`source scripts/install.sh`

This will take care of installing all required dependencies.
If you have trouble during the installation, try running each line of the `scripts/install.sh` file separately (one by one) in your shell.

The only required dependency is the latest Conda package manager, which you can download with the Anaconda Python distribution [here](https://www.anaconda.com/distribution/).

After that, you are all set up.

### Training

First, you need to download the data and do some preprocessing. To do this, run:

`python manage.py preprocess --dataset <DATASET_NAME>`

where `<DATASET_NAME>` must be `ZINC` or `PCBA`. At the moment, we support only these two.

Use `python manage.py preprocess --help` to see other useful options for preprocessing.

This will download the necessary files in the `DATA` folder, and will preprocess them as described in the paper.


After that, you can train the model. Suppose we want to train with the `ZINC` dataset, then you need to run:

`python manage.py train --dataset ZINC`

If you wish to train using a GPU, add the `--use_gpu` option.

Check out `python manage.py train --help` to see all the other hyperparameters you can change.

### Samples

20k SMILES samples obtained by the 4 variants of the model are placed in the folder SAMPLES.
