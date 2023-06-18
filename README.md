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
#### Train a new model
To train a new model, replace `path_data` and `path_valid_data` in `pipp/main.py`. Make sure you specify a model name when writing (saving) the model. Then from command line run:

```

python main.py --max-epoch --shot --test-way --test-shot --test-query --train-query --train-way

```
To train a new model, a few more package dependencies are required. See `import` statements in `pipp/main.py`. The code supports training on GPU.



