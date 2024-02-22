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

    # load lists
    # with open('example_data/false_neighbour_list.pkl', 'rb') as file:
    #    false_neighbour_list = pickle.load(file)
    # with open('example_data/true_neighbour_list.pkl', 'rb') as file:
    #    true_neighbour_list = pickle.load(file)
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
    false_transfer_knn_index = NNDescent(
        false_neighbour_embeddings_df, metric="euclidean"
    )

    true_transfer_knn_index = NNDescent(
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
        false_neighbour_neighbours, false_neighbour_distances = false_transfer_knn_index.query(
            example_df, k=250)
        false_neighbour_neighbours_cosine, false_neighbour_distances_cosine = false_transfer_knn_index_cosine.query(
            example_df, k=250)
        pynn_distances = false_neighbour_distances[:, neighbour_indexes]
        pynn_distances_cosine = false_neighbour_distances_cosine[:, neighbour_indexes]

        # sklearn distances
        example_array = example_df.values
        neighbour_embeddings = false_neighbour_embeddings_df.loc[neighbour_indexes]
        neighbour_array = neighbour_embeddings.values
        sklearn_distances = metrics.pairwise_distances(example_array, neighbour_array)

        new_tuple = tuple + (example_index, neighbour_indexes, pynn_distances, pynn_distances_cosine, sklearn_distances)
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
        true_neighbour_neighbours, true_neighbour_distances = true_transfer_knn_index.query(
            example_df, k=250)
        true_neighbour_neighbours_cosine, true_neighbour_distances_cosine = true_transfer_knn_index_cosine.query(
            example_df, k=250)
        pynn_distances = true_neighbour_distances[:, neighbour_indexes]
        pynn_distances_cosine = true_neighbour_distances_cosine[:, neighbour_indexes]

        # sklearn distances
        example_array = example_df.values
        neighbour_embeddings = true_neighbour_embeddings_df.loc[neighbour_indexes]
        neighbour_array = neighbour_embeddings.values
        sklearn_distances = metrics.pairwise_distances(example_array, neighbour_array)

        new_tuple = tuple + (example_index, neighbour_indexes, pynn_distances, pynn_distances_cosine, sklearn_distances)
        ultimate_true_list.append(new_tuple)

    # prepare data for plot
    false_transfer_distance_list_pynn = []
    false_transfer_distance_list_pynn_cosine = []
    false_transfer_distance_list_sklearn = []
    for element in ultimate_false_list:
        false_transfer_distance_list_pynn.extend((element[4][0]).tolist())
        false_transfer_distance_list_pynn_cosine.extend((element[5][0]).tolist())
        false_transfer_distance_list_sklearn.extend((element[6][0]).tolist())

    false_transfer_distance_df_sklearn = pd.DataFrame(false_transfer_distance_list_sklearn)
    false_transfer_distance_df_pynn = pd.DataFrame(false_transfer_distance_list_pynn)
    false_transfer_distance_df_pynn_cosine = pd.DataFrame(false_transfer_distance_list_pynn_cosine)

    true_transfer_distance_list_pynn = []
    true_transfer_distance_list_pynn_cosine = []
    true_transfer_distance_list_sklearn = []
    for element in ultimate_true_list:
        true_transfer_distance_list_pynn.extend((element[4][0]).tolist())
        true_transfer_distance_list_pynn_cosine.extend((element[5][0]).tolist())
        true_transfer_distance_list_sklearn.extend((element[6][0]).tolist())

    true_transfer_distance_df_sklearn = pd.DataFrame(true_transfer_distance_list_sklearn)
    true_transfer_distance_df_pynn = pd.DataFrame(true_transfer_distance_list_pynn)
    true_transfer_distance_df_pynn_cosine = pd.DataFrame(true_transfer_distance_list_pynn_cosine)

    # plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hist(false_transfer_distance_df_sklearn, bins=40, color='blue', edgecolor='black', linewidth=1.2)
    ax.set_title('False transfer | Distribution of ProtT5 distances with sklearn')
    ax.set_xlabel('Distance')
    ax.set_ylabel('Number of distances')
    ax.grid(True)
    ax.set_axisbelow(True)
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.hist(false_transfer_distance_df_pynn, bins=40, color='blue', edgecolor='black', linewidth=1.2)
    ax2.set_title('False transfer | Distribution of ProtT5 distances with PyNNDescent')
    ax2.set_xlabel('Distance')
    ax2.set_ylabel('Number of distances')
    ax2.grid(True)
    ax2.set_axisbelow(True)
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hist(true_transfer_distance_df_sklearn, bins=40, color='blue', edgecolor='black', linewidth=1.2)
    ax.set_title('True transfer | Distribution of ProtT5 distances with sklearn')
    ax.set_xlabel('Distance')
    ax.set_ylabel('Number of distances')
    ax.grid(True)
    ax.set_axisbelow(True)
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.hist(true_transfer_distance_df_pynn, bins=40, color='blue', edgecolor='black', linewidth=1.2)
    ax2.set_title('True transfer | Distribution of ProtT5 distances with PyNNDescent')
    ax2.set_xlabel('Distance')
    ax2.set_ylabel('Number of distances')
    ax2.grid(True)
    ax2.set_axisbelow(True)
    plt.show()

    min_length = min(len(true_transfer_distance_df_sklearn), len(false_transfer_distance_df_sklearn))

    # Truncate DataFrames to the length of the shortest one
    true_transfer_distance_df_sklearn = true_transfer_distance_df_sklearn.head(min_length)
    false_transfer_distance_df_sklearn = false_transfer_distance_df_sklearn.head(min_length)
    true_transfer_distance_df_pynn = true_transfer_distance_df_pynn.head(min_length)
    false_transfer_distance_df_pynn = false_transfer_distance_df_pynn.head(min_length)

    sklearn_ttest_result = scipy.stats.ttest_ind(true_transfer_distance_df_sklearn, false_transfer_distance_df_sklearn)
    pynn_ttest_result = scipy.stats.ttest_ind(true_transfer_distance_df_pynn, false_transfer_distance_df_pynn)
    sklearn_wilcoxon_result = scipy.stats.wilcoxon(true_transfer_distance_df_sklearn,
                                                   false_transfer_distance_df_sklearn)
    pynn_wilcoxon_result = scipy.stats.wilcoxon(true_transfer_distance_df_pynn, false_transfer_distance_df_pynn)
    #sklearn_wasserstein_result = scipy.stats.wasserstein_distance(true_transfer_distance_df_sklearn,
    #                                                              false_transfer_distance_df_sklearn)
    #pynn_wasserstein_result = scipy.stats.wasserstein_distance(true_transfer_distance_df_pynn,
    #                                                           false_transfer_distance_df_pynn)

    print('sklearn ttest result: ', sklearn_ttest_result.pvalue)
    print('pynn ttest result: ', pynn_ttest_result.pvalue)
    print('sklearn wilcoxon result: ', sklearn_wilcoxon_result.pvalue)
    print('pynn wilcoxon result: ', pynn_wilcoxon_result.pvalue)
    #print('sklearn wasserstein result: ', sklearn_wasserstein_result)
    #print('pynn wasserstein result: ', pynn_wasserstein_result)

    sns.set(rc={'figure.figsize': (14, 14)})
    sns.set_style("whitegrid", {'axes.grid': False})

    df = pd.DataFrame()
    df['ProtT5\ntrue transfer\nSklearn distances'] = true_transfer_distance_df_sklearn
    df['ProtT5\nfalse transfer\nSklearn distances'] = false_transfer_distance_df_sklearn
    df['ProtT5\ntrue transfer\nPyNNDescent (euclidean)\ndistances'] = true_transfer_distance_df_pynn
    df['ProtT5\nfalse transfer\nPyNNDescent (euclidean)\ndistances'] = false_transfer_distance_df_pynn
    df['ProtT5\ntrue transfer\nPyNNDescent (cosine)\ndistances'] = true_transfer_distance_df_pynn_cosine
    df['ProtT5\nfalse transfer\nPyNNDescent (cosine)\ndistances'] = false_transfer_distance_df_pynn_cosine

    ax = sns.boxplot(data=df, palette='Oranges', width=0.5)
    plt.show()

'''
If you want to compute and store protT5 embeddings, use call_protT5()
'''
# call_protT5()

'''
If you already have embeddings that are stored in a .pth file, use plot_stored_embeddings()
'''
# plot_stored_embeddings()

'''
If you want to compute the distances of false/true transfer prototype examples, use get_neighbor_distances_of_examples()
'''
get_neighbor_distances_of_examples()



