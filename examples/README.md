# Relevant files and scripts:
1. **Main Notebooks:**
	* `examples/PXD014777_HeLa.ipynb`: Main notebook to conduct PIPP to measure missing values.
	* `examples/PXD014777_HeLa_multi_test.ipynb`: Adaption of previous notebook just for running it with multiple configurations.
	* `examples/PXD014777_HeLA_Yeast_Ecoli.ipynb`: Main notebook to conduct PIPP to measure false discovery rate.
	* `examples/PXD014777_HeLA_Yeast_Ecoli_multi_test.ipynb`: Adaption of previous notebook just for running it with multiple configurations.
2. **Simple examples that show basic usage offered by the creators of the PLMs:**
	* `examples/esm_example.py`
	* `examples/protTrans_example.py`
3. **Scripts to generate embeddings and conduct the neighbour / species separation experiment outside the notebooks:**
	* `examples/generate_ProtT5_embeddings.py`: Generate and plot embeddings with ProtT5.
	* `examples/generate_ESM_embeddings.py`: Generate embeddings with multiple configurations of ESM-2.
	* `examples/species_separation_experiment.py`: Conduct the species separation experiment to compare ESM-2 and ProtT5.
4. **Visualization of False Discovery Rate:**
	* `figures/PIPP_with_ESM_visualization.ipynb`: Plot different views of false discovery rate.
      requires a .csv file as input like `examples/example_data/fdr_multi_test_esm2_t6_8M_UR50D_layer6.csv`
5. **Data:**
	* `examples/example_data/all_prototypes.npz`: Precomputed prototypes for `examples/PXD014777_HeLA_Yeast_Ecoli.ipynb`
       and `examples/PXD014777_HeLA_Yeast_Ecoli_multi_test.ipynb`.
	* `examples/example_data/all_prototypes_H.npz`: Precomputed prototypes for `examples/PXD014777_HeLa.ipynb` and
       `examples/PXD014777_HeLa_multi_test.ipynb`.
	* `examples/example_data/false_positives_and_neighbours.pkl`: Preselected false positives and their precomputed 5
      closest neighbours. Used for `examples/species_separation_experiment.py`.
	* `examples/example_data/true_positives_and_neighbours.pkl`: Preselected true positives and their precomputed 5
      closest neighbours. Used for `examples/species_separation_experiment.py`.
    * `examples/example_data/fdr_multi_test_esm2_t6_8M_UR50D_layer6.csv`: FDR test results of 7 test runs 
      (high anchor number were tested fewer times due to their computation time). The model esm2_t6_8M_UR50D was used 
      with the 6th representation layer. Can be used as input for 
      `figures/PIPP_with_ESM_visualization.ipynb`
    * `examples/example_data/fdr_multi_test_esm2_t33_650M_UR50D_layer_6.csv`: FDR test results of 6 test runs
      (high anchor number were tested fewer times due to their computation time). The model esm2_t33_650M_UR50D was
      used with the 6th representation layer. Can be used as input for 
      `figures/PIPP_with_ESM_visualization.ipynb`
    * `examples/example_data/embeddings_esm2_t6_8M_UR50D_layer6_datasetHeLa.npy`: precomputed ESM-2 embeddings with
      the model esm2_t6_8M_UR50D with the 6th representation layer for the HeLa-only dataset/notebook.
    * `examples/example_data/embeddings_esm2_t6_8M_UR50D_layer6_datasetHeLa_Yeast_Ecoli.npy`: precomputed ESM-2 embeddings with
      the model esm2_t6_8M_UR50D with the 6th representation layer for the HeLa_Yeast_Ecoli dataset/notebook.
    * `examples/example_data/embeddings_esm2_t33_650M_UR50D_layer6_datasetHeLa.npy`: precomputed ESM-2 embeddings with
      the model esm2_t33_650M_UR50D with the 6th representation layer for the HeLa-only dataset/notebook.
    * `examples/example_data/embeddings_esm2_t33_650M_UR50D_layer6_datasetHeLa_Yeast_Ecoli.npy`: precomputed ESM-2 embeddings with
      the model esm2_t33_650M_UR50D with the 6th representation layer for the HeLa_Yeast_Ecoli dataset/notebook.