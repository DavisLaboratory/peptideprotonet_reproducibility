from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
import numpy as np
import pandas as pd
import random
import umap
import matplotlib.pyplot as plt
from sklearn import metrics
import pickle
from pynndescent import NNDescent
import scipy
import seaborn as sns

'''
This script is used for the following tasks:

1. Generate and store ProtT5 embeddings with the function call_protT5().
   This function expects the file "example_data/all_prototypes.npz" (contains 87605 prototypes).
   This function stores ProtT5 embeddings at 'example_data/protT5_embeddings_df.pkl'.
2. Plot precomputed ProtT5 embeddings with the function plot_stored_embeddings().
   This function expects the file 'example_data/protT5_embeddings_df.pkl' 
3. Conduct the neighbour experiment with the function get_neighbor_distances_of_examples().
   This function expects the files 'example_data/final_false_list.pkl' and 'example_data/final_true_list.pkl'.
   
Select at the very bottom of this script which functions you want to run.
'''

def get_protT5_embedding(device, tokenizer, model, sequences):
    sequence_representations = []

    # Iterate through the list in bulks because the execution is more stable that way.
    bulk_size = 500
    for i in range(0, len(sequences), bulk_size):
        print("i:", i)
        bulk = sequences[i:i + bulk_size]

        # replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
        bulk = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in bulk]

        # tokenize sequences and pad up to the longest sequence in the batch
        ids = tokenizer(bulk, add_special_tokens=True, padding="longest")

        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        # generate embeddings
        with torch.no_grad():
            embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

        # derive a single representation for each sequence
        for embedding in embedding_repr.last_hidden_state:
            representation = embedding.mean(dim=0)
            sequence_representations.append(representation)

    return sequence_representations

def call_protT5():

    # Setup steps of protT5:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load the tokenizer
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

    # Load the model
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)

    # only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
    if device == torch.device("cpu"):
        model.to(torch.float32)

    # Load prototypes and prepare sequences as a list
    prototypes = np.load('example_data/all_prototypes.npz')
    prototypes = dict(prototypes)

    # Get a random sample indices
    subset_size = 86000
    sampled_indices = random.sample(range(len(prototypes['Sequence'])), subset_size)

    # Create a new dictionary with entries at the sampled indices
    random_subset_prototypes = {key: [prototypes[key][i] for i in sampled_indices] for key in prototypes}

    prototype_sequences = random_subset_prototypes['Sequence']

    # Get protT5 embeddings
    embeddings = get_protT5_embedding(device, tokenizer, model, prototype_sequences)

    # Connect embeddings with PrecursorID and species
    random_subset_prototypes_df = pd.DataFrame(random_subset_prototypes)
    random_subset_prototypes_df['protT5_embeddings'] = embeddings

    # Store dataframe
    random_subset_prototypes_df.to_pickle('example_data/protT5_embeddings_df.pkl')

def plot_stored_embeddings():
    # Load embeddings
    embedding_df = pd.read_pickle('example_data/protT5_embeddings_df.pkl')

    # Reduce with UMAP
    embeddings = embedding_df['protT5_embeddings']

    correct_embeddings = []
    tensors = []

    for tensor_array in embeddings:
        value_array = np.array([tensor.item() for tensor in tensor_array])
        correct_embeddings.append(value_array)
        tensors.append(tensor_array)

    reducer = umap.UMAP(metric='cosine')
    umap_embeddings = reducer.fit_transform(tensors)
    umap1 = umap_embeddings[:, 0:1]
    umap2 = umap_embeddings[:, 1:2]

    embedding_df['umap1'] = umap1
    embedding_df['umap2'] = umap2

    # Prepare plot of general UMAP-embedding with species.
    colors = ['red', 'blue', 'green']
    color_map = {species: color for species, color in zip(('HeLa', 'Yeast', 'Ecoli'), colors)}
    species_size = {'HeLa': .1, 'Yeast': .4, 'Ecoli': 1.4}
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_title('ProtT5 embedding | species')

    # Plot general UMAP-embedding with species.
    for species, color in color_map.items():
        prototypes_with_current_species = embedding_df[embedding_df['Species'] == species]
        umap1 = prototypes_with_current_species['umap1']
        umap2 = prototypes_with_current_species['umap2']
        ax.scatter(umap1, umap2, c=color, s=species_size[species], label=species, alpha=0.6)

    ax.legend()
    plt.show()

def get_neighbor_distances_of_examples():

    with open('example_data/final_false_list.pkl', 'rb') as file:
        final_false_list = pickle.load(file)
    with open('example_data/final_true_list.pkl', 'rb') as file:
        final_true_list = pickle.load(file)

    false_neighbour_list = []
    true_neighbour_list = []

    for row in final_false_list:
        false_neighbour_list.append(row[0])
        for neighbour in row[1]:
            false_neighbour_list.append(neighbour)

    for row in final_true_list:
        true_neighbour_list.append(row[0])
        for neighbour in row[1]:
            true_neighbour_list.append(neighbour)

    # prepare protT5
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load the tokenizer
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

    # Load the model
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)

    # only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
    if device == torch.device("cpu"):
        model.to(torch.float32)

    # Prepare lists of examples and their neighbors:
    false_sequences = [t[1] for t in false_neighbour_list]
    true_sequences = [t[1] for t in true_neighbour_list]

    # Get protT5 embeddings
    false_neighbour_embeddings = get_protT5_embedding(device, tokenizer, model, false_sequences)
    true_neighbour_embeddings = get_protT5_embedding(device, tokenizer, model, true_sequences)

    # transform list of embeddings to dataframe
    false_neighbour_embeddings_df = pd.DataFrame(false_neighbour_embeddings)
    true_neighbour_embeddings_df = pd.DataFrame(true_neighbour_embeddings)

    # set up Pynndescent neighbour graphs and retrieve distances
    false_transfer_knn_index_euclidean = NNDescent(
        false_neighbour_embeddings_df, metric="euclidean"
    )

    true_transfer_knn_index_euclidean = NNDescent(
        true_neighbour_embeddings_df, metric="euclidean"
    )

    false_transfer_knn_index_cosine = NNDescent(
        false_neighbour_embeddings_df, metric="cosine"
    )

    true_transfer_knn_index_cosine = NNDescent(
        true_neighbour_embeddings_df, metric="cosine"
    )

    # match embeddings, neighbours and distances to their original PrecursorIDs and sequences
    ultimate_false_list = []
    for tuple in final_false_list:
        example_index = false_neighbour_list.index(tuple[0])
        neighbour_indexes = []
        for neighbour in tuple[1]:
            neighbour_indexes.append(false_neighbour_list.index(neighbour))

        # PyNNDescent distances
        example = false_neighbour_embeddings_df.loc[example_index]
        example_df = pd.DataFrame([example])
        false_neighbour_neighbours_euclidean, false_neighbour_distances_euclidean = false_transfer_knn_index_euclidean.query(
            example_df, k=250)
        false_neighbour_neighbours_cosine, false_neighbour_distances_cosine = false_transfer_knn_index_cosine.query(
            example_df, k=250)
        pynn_distances_euclidean = false_neighbour_distances_euclidean[:, neighbour_indexes]
        pynn_distances_cosine = false_neighbour_distances_cosine[:, neighbour_indexes]

        # sklearn distances
        example_array = example_df.values
        neighbour_embeddings = false_neighbour_embeddings_df.loc[neighbour_indexes]
        neighbour_array = neighbour_embeddings.values
        sklearn_distances = metrics.pairwise_distances(example_array, neighbour_array)

        new_tuple = tuple + (example_index, neighbour_indexes, pynn_distances_euclidean, pynn_distances_cosine, sklearn_distances)
        ultimate_false_list.append(new_tuple)

    ultimate_true_list = []
    for tuple in final_true_list:
        example_index = true_neighbour_list.index(tuple[0])
        neighbour_indexes = []
        for neighbour in tuple[1]:
            neighbour_indexes.append(true_neighbour_list.index(neighbour))

        # PyNNDescent distances
        example = true_neighbour_embeddings_df.loc[example_index]
        example_df = pd.DataFrame([example])
        true_neighbour_neighbours_euclidean, true_neighbour_distances_euclidean = true_transfer_knn_index_euclidean.query(
            example_df, k=250)
        true_neighbour_neighbours_cosine, true_neighbour_distances_cosine = true_transfer_knn_index_cosine.query(
            example_df, k=250)
        pynn_distances_euclidean = true_neighbour_distances_euclidean[:, neighbour_indexes]
        pynn_distances_cosine = true_neighbour_distances_cosine[:, neighbour_indexes]

        # sklearn distances
        example_array = example_df.values
        neighbour_embeddings = true_neighbour_embeddings_df.loc[neighbour_indexes]
        neighbour_array = neighbour_embeddings.values
        sklearn_distances = metrics.pairwise_distances(example_array, neighbour_array)

        new_tuple = tuple + (example_index, neighbour_indexes, pynn_distances_euclidean, pynn_distances_cosine, sklearn_distances)
        ultimate_true_list.append(new_tuple)

    # prepare data for plot
    false_transfer_distance_list_pynn_euclidean = []
    false_transfer_distance_list_pynn_cosine = []
    false_transfer_distance_list_sklearn = []
    for element in ultimate_false_list:
        false_transfer_distance_list_pynn_euclidean.extend((element[4][0]).tolist())
        false_transfer_distance_list_pynn_cosine.extend((element[5][0]).tolist())
        false_transfer_distance_list_sklearn.extend((element[6][0]).tolist())

    false_transfer_distance_df_sklearn = pd.DataFrame(false_transfer_distance_list_sklearn)
    false_transfer_distance_df_pynn_euclidean = pd.DataFrame(false_transfer_distance_list_pynn_euclidean)
    false_transfer_distance_df_pynn_cosine = pd.DataFrame(false_transfer_distance_list_pynn_cosine)

    true_transfer_distance_list_pynn_euclidean = []
    true_transfer_distance_list_pynn_cosine = []
    true_transfer_distance_list_sklearn = []
    for element in ultimate_true_list:
        true_transfer_distance_list_pynn_euclidean.extend((element[4][0]).tolist())
        true_transfer_distance_list_pynn_cosine.extend((element[5][0]).tolist())
        true_transfer_distance_list_sklearn.extend((element[6][0]).tolist())

    true_transfer_distance_df_sklearn = pd.DataFrame(true_transfer_distance_list_sklearn)
    true_transfer_distance_df_pynn_euclidean = pd.DataFrame(true_transfer_distance_list_pynn_euclidean)
    true_transfer_distance_df_pynn_cosine = pd.DataFrame(true_transfer_distance_list_pynn_cosine)

    min_length = min(len(true_transfer_distance_df_sklearn), len(false_transfer_distance_df_sklearn))

    # Truncate DataFrames to the length of the shortest one
    true_transfer_distance_df_sklearn = true_transfer_distance_df_sklearn.head(min_length)
    false_transfer_distance_df_sklearn = false_transfer_distance_df_sklearn.head(min_length)
    true_transfer_distance_df_pynn_euclidean = true_transfer_distance_df_pynn_euclidean.head(min_length)
    false_transfer_distance_df_pynn_euclidean = false_transfer_distance_df_pynn_euclidean.head(min_length)

    sklearn_ttest_result = scipy.stats.ttest_ind(true_transfer_distance_df_sklearn, false_transfer_distance_df_sklearn)
    pynn_ttest_result = scipy.stats.ttest_ind(true_transfer_distance_df_pynn_euclidean, false_transfer_distance_df_pynn_euclidean)
    sklearn_wilcoxon_result = scipy.stats.wilcoxon(true_transfer_distance_df_sklearn,
                                                   false_transfer_distance_df_sklearn)
    pynn_wilcoxon_result = scipy.stats.wilcoxon(true_transfer_distance_df_pynn_euclidean, false_transfer_distance_df_pynn_euclidean)

    print('sklearn ttest result: ', sklearn_ttest_result.pvalue)
    print('pynn ttest result: ', pynn_ttest_result.pvalue)
    print('sklearn wilcoxon result: ', sklearn_wilcoxon_result.pvalue)
    print('pynn wilcoxon result: ', pynn_wilcoxon_result.pvalue)

    sns.set(rc={'figure.figsize': (14, 14)})
    sns.set_style("whitegrid", {'axes.grid': False})

    df = pd.DataFrame()
    df['true transfer\nSklearn distances'] = true_transfer_distance_df_sklearn
    df['false transfer\nSklearn distances'] = false_transfer_distance_df_sklearn
    df['true transfer\nPyNNDescent (euclidean)\ndistances'] = true_transfer_distance_df_pynn_euclidean
    df['false transfer\nPyNNDescent (euclidean)\ndistances'] = false_transfer_distance_df_pynn_euclidean
    df['true transfer\nPyNNDescent (cosine)\ndistances'] = true_transfer_distance_df_pynn_cosine
    df['false transfer\nPyNNDescent (cosine)\ndistances'] = false_transfer_distance_df_pynn_cosine

    df.to_csv('example_data/neighbour_experiment_model_protT5.csv', index=False)

    ax = sns.boxplot(data=df, palette='Oranges', width=0.5)
    plt.title(f'Distances between true/false positives and their respective neighbours\nwith model ProtT5',
              fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=20)
    plt.show()

'''
If you want to compute and store protT5 embeddings, use call_protT5()
'''
generate_and_store_embeddings = False
if generate_and_store_embeddings:
    call_protT5()

'''
If you already have embeddings that are stored in a .pth file, use plot_stored_embeddings()
'''
plot_embeddings = True
if plot_embeddings:
    plot_stored_embeddings()

'''
If you want to compute the distances of false/true transfer prototype examples, use get_neighbor_distances_of_examples()
'''
conduct_neighbour_experiment = False
if conduct_neighbour_experiment:
    get_neighbor_distances_of_examples()



