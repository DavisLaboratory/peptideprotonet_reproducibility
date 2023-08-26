# Peptide identity propagation and match-between-runs by few-shot learning
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8051947.svg)](https://doi.org/10.5281/zenodo.8051947)

Code and notebooks to reproduce figures and results for the manuscript **PIPP: Improving peptide identity propagation using neural networks** 

![pipp_gabstract_github](https://github.com/DavisLaboratory/peptideprotonet_reproducibility/assets/7257233/341527f5-e055-4eba-aeae-e7fc61cacc54)

PIPP is a deep learning framework for match-between-run in DDA PASEF data quantified by MaxQuant.
We have trained a deep neural network model which learns an embedding of MS1 features of peptide identifications quantified in two large-scale DDA-PASEF datasets up to the date, namely PXD019086 and PXD010012 datasets. The model is learnt by a novel modification of Prototypical Networks, which is a few-shot learning classification algorithm. The pre-trained model is used for peptide identity propagation to match identifications between runs, increase protein coverage and improve data completeness.

The pre-trained model, train/test splits and pre-computed embeddings can be downloaded from [Zenodo](https://zenodo.org/record/8051947). 


### Installation
Change working directory to "peptideprotonet_reproducibility", then (with your virtual environment activated) execute:

```
pip install .
```
This will install the PIPP library, including all the dependencies needed to use the library and run the notebooks in ```/examples/```.



### Usage

#### Use the pre-trained model
```
```

### Train a new model
To train a new model, replace `path_data` and `path_valid_data` in `pipp/main.py`. Make sure you specify a model name when writing (saving) the model. Then from the command line, run:

```

# define the number of shots for train and test e.g. 0-shot, 1-shot, 5-shot etc
n_shot_train
n_shot_test

# define the number of classes for train and test e.g. 2-way, 3-way, 5-way classification etc
n_test
n_train

# define the number of support instances (query) for train and test.
# These are number of instances that are selected as instances in the "support set"
# support set is used to compute the prototype at each batch/round of train and test
nq_test
nq_train



python main.py --max-epoch 300
               --shot n_shot_train
               --test-way n_test
               --test-shot n_shot_test
               --test-query nq_test
               --train-query nq_train
               --train-way n_train

```
If you wish to train a new model, a few more package dependencies are required. See below or `import` statements in `pipp/main.py`. The code supports training on the GPU.

###### Additional dependencies required to train a new model
- \__future\__
- argparse
- pickle
- learn2learn
- sklearn


Any problems? Let us know by openning a new issue!
