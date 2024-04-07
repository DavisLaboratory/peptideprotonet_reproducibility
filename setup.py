from setuptools import setup, find_packages

setup(
    name='pipp', 
    version='0.1',
    description='Python library for the model proposed in \"PIPP: Improving peptide identity propagation using neural networks.\"',
    packages=find_packages(),

    install_requires=[
        'torch>=2.1.2',
        'pynndescent>=0.5.11',
        'numpy>=1.26.3',
        'scikit-learn>=1.3.2',
        'matplotlib>=3.8.2',
        'seaborn>=0.13.1',
        'pandas>=2.1.4',
        'umap-learn>=0.5.5',
        'fair-esm>=2.0.0',           # for esm
        'transformers>=4.36.2',      # for protT5
        'sentencepiece>=0.1.99',     # for protT5
        'protobuf>=4.25.2',          # for protT5
        'scipy>=1.11.4',
        'scanpy>=1.9.8',
        'plotly>=5.20.0'
    ]

)
