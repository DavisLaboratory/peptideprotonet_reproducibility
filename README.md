# Peptide identity propagation and match-between-runs by few-shot learning
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8051947.svg)](https://doi.org/10.5281/zenodo.8051947)

Code and notebooks to reproduce figures and results for the manuscript **PIPP: Improving peptide identity propagation using neural networks** 

![pipp_gabstract_github](https://github.com/DavisLaboratory/peptideprotonet_reproducibility/assets/7257233/341527f5-e055-4eba-aeae-e7fc61cacc54)

PIPP is a deep learning framework for match-between-run in DDA PASEF data quantified by MaxQuant.
Pre-trained model, train/test splits and pre-computed embeddings can be downloaded from [Zenodo](https://zenodo.org/record/8051947) 

### Requierments
```
python
torch
```
### Usage

#### Use the pre-trained model
```
```

### Train a new model
To train a new model, replace `path_data` and `path_valid_data` in `pipp/main.py`. Make sure you specify a model name when writing (saving) the model. Then from command line run:

```

# define number of shots for train and test e.g. 0-shot, 1-shot, 5-shot etc
n_shot_train
n_shot_test

# define number of classes for train and test e.g. 2-way, 3-way, 5-way classification etc
n_test
n_train

# define number of support instances (query) for train and test.
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
To train a new model, a few more package dependencies are required. See below or `import` statements in `pipp/main.py`. The code supports training on GPU.

###### Additional dependencies required to train a new model
- \__future\__
- argparse
- pickle
- learn2learn
- sklearn


Any problems? Let us know by openning a new issue!
