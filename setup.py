from setuptools import setup, find_packages

setup(
    name='pipp', 
    version='0.1',
    description='Python library for the model proposed in \"PIPP: Improving peptide identity propagation using neural networks.\"',
    packages=find_packages(),

    install_requires=[
        'torch',
        'pynndescent',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'pandas',
        'umap-learn',
        'fair-esm',         # for esm
        'transformers',     # for protT5
        'sentencepiece',    # for protT5
        'protobuf'          # for protT5
    ]

)
