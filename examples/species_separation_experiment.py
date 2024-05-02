import pandas as pd
from pynndescent import NNDescent
import esm
import torch
import pickle
from sklearn import metrics
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
from transformers import T5Tokenizer, T5EncoderModel
import re

'''
This script is used to conduct the neighbour / species separation experiment with ESM-2 and ProtT5.
It requires the following precomputed files that contain false/true positives and their neighbours:
    - 'example_data/false_positives_and_neighbours.pkl'
    - 'example_data/true_positives_and_neighbours.pkl'
Both files are alternatively created by running PXD014777_HeLA_Yeast_Ecoli.ipynb with setting conduct_neighbour_experiment = False.
It creates the output file 'example_data/neighbour_experiment_model_{model_name}_layer{representation_layer}.csv'
and plots several visualizations.
'''

def get_esm_embedding(batch_converter, model, alphabet, data, representation_layer):
    """
    Main function to retrieve ESM-2 embeddings after setting up a model, alphabet and batch_converter.

    Parameters
    ----------

    batch_converter
        ESM-internal component to process input in batches.

    model
        ESM-2 model.

    alphabet
        ESM-2 model-specific alphabet.

    data
        List of tuples in the form ('PrecursorID',  'Sequence').

    representation_layer
        The n-th ESM-2 layer from which to retrieve the representations.

    Returns
    -------
        List of representations.

    """

    sequence_representations = []

    # Iterate through the list in bulks because the execution is more stable that way.
    # bulk_size of 500 is not necessarily the best size.
    bulk_size = 500
    for i in range(0, len(data), bulk_size):
        print("i:", i)
        bulk = data[i:i + bulk_size]

        # Use batch converter as done in ESM-2 example.
        batch_labels, batch_strs, batch_tokens = batch_converter(bulk)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        # Extract per-residue representations (on CPU) as done in ESM-2 example.
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[representation_layer], return_contacts=True)
        token_representations = results["representations"][representation_layer]

        # Generate per-sequence representations via averaging as done in ESM-2 example.
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))

    return sequence_representations


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

def neighbour_experiment(
        model_name,
        representation_layer,
        visualize,
        final_false_list,
        final_true_list,
        false_neighbour_list,
        true_neighbour_list):

    if model_name == "ProtT5":

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

    else:

        # prepare ESM-2
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        batch_converter = alphabet.get_batch_converter()
        model.eval()  # disables dropout for deterministic results

        # compute ESM embeddings
        false_neighbour_embeddings = get_esm_embedding(
            batch_converter,
            model,
            alphabet,
            false_neighbour_list,
            representation_layer=representation_layer)

        true_neighbour_embeddings = get_esm_embedding(
            batch_converter,
            model,
            alphabet,
            true_neighbour_list,
            representation_layer=representation_layer)

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

    '''
    Here, embeddings and neighbours are matched to their original PrecursorIDs and sequences in false_positives_and_neighbours and
    true_positives_and_neighbours. Three different types of distances between false/true positives and their neighbours are computed:
        - pynn_distances_euclidean: 
        - pynn_distances_cosine
        - sklearn_distances
    '''
    final_false_list_with_distances = []
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

        new_tuple = tuple + (
        example_index, neighbour_indexes, pynn_distances_euclidean, pynn_distances_cosine, sklearn_distances)
        final_false_list_with_distances.append(new_tuple)

    final_true_list_with_distances = []
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

        new_tuple = tuple + (
        example_index, neighbour_indexes, pynn_distances_euclidean, pynn_distances_cosine, sklearn_distances)
        final_true_list_with_distances.append(new_tuple)

    # prepare data for plot
    false_transfer_distance_list_pynn_euclidean = []
    false_transfer_distance_list_pynn_cosine = []
    false_transfer_distance_list_sklearn = []
    for element in final_false_list_with_distances:
        false_transfer_distance_list_pynn_euclidean.extend((element[4][0]).tolist())
        false_transfer_distance_list_pynn_cosine.extend((element[5][0]).tolist())
        false_transfer_distance_list_sklearn.extend((element[6][0]).tolist())

    false_transfer_distance_df_sklearn = pd.DataFrame(false_transfer_distance_list_sklearn)
    false_transfer_distance_df_pynn_euclidean = pd.DataFrame(false_transfer_distance_list_pynn_euclidean)
    false_transfer_distance_df_pynn_cosine = pd.DataFrame(false_transfer_distance_list_pynn_cosine)

    true_transfer_distance_list_pynn_euclidean = []
    true_transfer_distance_list_pynn_cosine = []
    true_transfer_distance_list_sklearn = []
    for element in final_true_list_with_distances:
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
    true_transfer_distance_df_pynn_cosine = true_transfer_distance_df_pynn_cosine.head(min_length)
    false_transfer_distance_df_pynn_cosine = false_transfer_distance_df_pynn_cosine.head(min_length)

    sklearn_ttest_result = scipy.stats.ttest_ind(true_transfer_distance_df_sklearn, false_transfer_distance_df_sklearn)
    pynn_ttest_result_euclidean = scipy.stats.ttest_ind(true_transfer_distance_df_pynn_euclidean,
                                                        false_transfer_distance_df_pynn_euclidean)
    pynn_ttest_result_cosine = scipy.stats.ttest_ind(true_transfer_distance_df_pynn_cosine,
                                                     false_transfer_distance_df_pynn_cosine)
    sklearn_wilcoxon_result = scipy.stats.wilcoxon(true_transfer_distance_df_sklearn,
                                                   false_transfer_distance_df_sklearn)
    pynn_wilcoxon_result_euclidean = scipy.stats.wilcoxon(true_transfer_distance_df_pynn_euclidean,
                                                          false_transfer_distance_df_pynn_euclidean)
    pynn_wilcoxon_result_cosine = scipy.stats.wilcoxon(true_transfer_distance_df_pynn_cosine,
                                                       false_transfer_distance_df_pynn_cosine)
    # sklearn_wasserstein_result = scipy.stats.wasserstein_distance(true_transfer_distance_df_sklearn, false_transfer_distance_df_sklearn)
    # pynn_wasserstein_result = scipy.stats.wasserstein_distance(true_transfer_distance_df_pynn_euclidean, false_transfer_distance_df_pynn_euclidean)
    print('MODEL:', model_name, ' CHOSEN LAYER:', representation_layer)
    print('sklearn ttest result: ', sklearn_ttest_result.pvalue)
    print('pynn cosine ttest result: ', pynn_ttest_result_cosine.pvalue)
    print('pynn ttest result: ', pynn_ttest_result_euclidean.pvalue)
    print('sklearn wilcoxon result: ', sklearn_wilcoxon_result.pvalue)
    print('pynn cosine wilcoxon result: ', pynn_wilcoxon_result_cosine.pvalue)
    print('pynn wilcoxon result: ', pynn_wilcoxon_result_euclidean.pvalue)
    # print('sklearn wasserstein result: ', sklearn_wasserstein_result)
    # print('pynn wasserstein result: ', pynn_wasserstein_result)

    sns.set(rc={'figure.figsize': (14, 14)})
    sns.set_style("whitegrid", {'axes.grid': False})

    df = pd.DataFrame()
    df['true_sklearn'] = true_transfer_distance_df_sklearn
    df['false_sklearn'] = false_transfer_distance_df_sklearn
    df['true_pynn_euclidean'] = true_transfer_distance_df_pynn_euclidean
    df['false_pynn_euclidean'] = false_transfer_distance_df_pynn_euclidean
    df['true_pynn_cosine'] = true_transfer_distance_df_pynn_cosine
    df['false_pynn_cosine'] = false_transfer_distance_df_pynn_cosine

    df.to_csv(f'example_data/neighbour_experiment_model_{model_name}_layer{representation_layer}.csv', index=False)

    tt_mean = df['true_pynn_euclidean'].mean()
    tt_median = df['true_pynn_euclidean'].median()
    ft_mean = df['false_pynn_euclidean'].mean()
    ft_median = df['false_pynn_euclidean'].median()
    mean_diff = tt_mean - ft_mean
    median_diff = tt_median - ft_median

    new_row_for_df = [
        model_name,
        representation_layer,
        pynn_ttest_result_euclidean.pvalue,
        pynn_wilcoxon_result_euclidean.pvalue,
        mean_diff,
        median_diff]

    if visualize:
        new_column_names = {'true_sklearn': 'true transfer\nSklearn distances',
                            'false_sklearn': 'false transfer\nSklearn distances',
                            'true_pynn_euclidean': 'true transfer\nPyNNDescent\n(euclidean)\ndistances',
                            'false_pynn_euclidean': 'false transfer\nPyNNDescent\n(euclidean)\ndistances',
                            'true_pynn_cosine': 'true transfer\nPyNNDescent\n(cosine)\ndistances',
                            'false_pynn_cosine': 'false transfer\nPyNNDescent\n(cosine)\ndistances'}

        df.rename(columns=new_column_names, inplace=True)

        ax = sns.boxplot(data=df, palette='Reds', width=0.5)
        plt.title(f'Distances between true/false positives and their respective neighbours\nwith model {model_name} '
                  f'and layer {representation_layer}', fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=20)
        plt.show()

    return new_row_for_df

'''
Here, two lists are loaded:
    - false_positives_and_neighbours: list of tuples (false positive, 5 closest neighbours of false positive)
      a false positive is a precursor that was identified with a prototype with no 
      matching species. False positives are given here as a tuple (precursorID, sequence).
      The 5 closest neighbours of the false positive are given as a list of tuples (precursorID, sequence) with one
      tuple for each neighbour. 
    - true_positives_and_neighbours: same structure as false_positives_and_neighbours, but with a true positive and its 5 closest neighbours. 
      A true positive is a precursor that was identified with a prototype with matching species.
'''
with open('example_data/false_positives_and_neighbours.pkl', 'rb') as file:
    false_positives_and_neighbours = pickle.load(file)
with open('example_data/true_positives_and_neighbours.pkl', 'rb') as file:
    true_positives_and_neighbours = pickle.load(file)

'''
All false positives and their neighbours are put in a the common list false_neighbour_list which will be the input for
ESM-2. The ESM-2 embeddings are later matched again to the the false_positives_and_neighbours. The same is applied to the 
true_neighbour_list and true_positives_and_neighbours. This step was not necessary, instead false_positives_and_neighbours and true_positives_and_neighbours
could have been used with small changes as input for ESM-2.
'''
false_neighbour_list = []
true_neighbour_list = []

for row in false_positives_and_neighbours:
    false_neighbour_list.append(row[0])
    for neighbour in row[1]:
        false_neighbour_list.append(neighbour)

for row in true_positives_and_neighbours:
    true_neighbour_list.append(row[0])
    for neighbour in row[1]:
        true_neighbour_list.append(neighbour)

'''
The following ESM-2 models are available and are downloaded in the first use: 
    - "esm2_t6_8M_UR50D": 6 layers, 8 million parameters, 320-dim embedding (computation time for 80,000 prototypes: 10 minutes)
    - "esm2_t12_35M_UR50D": 12 layers, 35 million parameters, 480-dim embedding
    - "esm2_t30_150M_UR50D": 30 layers, 150 million parameters, 640-dim embedding
    - "esm2_t33_650M_UR50D": 33 layers, 650 million parameters, 1280-dim embedding (computation time for 80,000 prototypes: 10 hours)
    - "esm2_t36_3B_UR50D": : 36 layers, 3 billion parameters, 2560-dim embedding, 6GB size
    - "esm2_t48_15B_UR50D": 48 layers, 15 billion parameters, 5120-dim embedding, 30GB size
In the model names, the number after the "t" stands for the total layer number.
The next number stands for the parameter number.
"UR50D" stands for the used training dataset of the ESM-2 model which is UniRef 50.
'''
model_names = ["esm2_t6_8M_UR50D", "ProtT5"]

'''
This specifies the ESM-2 layer from which to retrieve the embeddings. Lower layers between 4 and 6 achieved the
best results. 
'''
representation_layers = [4, 5, 6]
visualize = True
df_all = pd.DataFrame(columns=['model_name', 'chosen_layer', 't_test', 'wsr_test', 'diff_mean', 'diff_median'])
for model_name in model_names:

    if model_name == "ProtT5":
        new_row = neighbour_experiment(
            model_name,
            24,
            visualize,
            false_positives_and_neighbours,
            true_positives_and_neighbours,
            false_neighbour_list,
            true_neighbour_list)

        df_all.loc[len(df_all.index)] = new_row
    else:
        for representation_layer in representation_layers:

            new_row = neighbour_experiment(
                model_name,
                representation_layer,
                visualize,
                false_positives_and_neighbours,
                true_positives_and_neighbours,
                false_neighbour_list,
                true_neighbour_list)

            df_all.loc[len(df_all.index)] = new_row


grouped = df_all.groupby('model_name')

if visualize:
    titles = ['Difference of means of True Positives and False Positives to the neighbours.\nShown for chosen representation layer and grouped by total layer number.',
              'Difference of medians of True Positives and False Positives to the neighbours.\nShown for chosen representation layer and grouped by total layer number.',
              't-test on the distributions of True Positives and False Positives to the neighbours.\nShown for chosen representation layer and grouped by total layer number.',
              'Wilcoxon-signed-rank test on the distributions of True Positives and False Positives to the neighbours.\nShown for chosen representation layer and grouped by total layer number.']
    ylabels = ['Difference of means (TP - FP)',
               'Difference of medians (TP - FP)',
               'p-values',
               'p-values']
    categories = ['diff_mean',
                'diff_median',
                't_test',
                'wsr_test']

    for i in range(len(titles)):

        # Create a plot
        plt.figure(figsize=(18, 10))

        # Plot each group separately
        for model_name, group in grouped:
            if model_name == 'ProtT5':
                plt.plot(
                    group['chosen_layer'],
                    group[categories[i]],
                    marker='o',
                    markersize=15,
                    label=f'ProtT5'
                )
            else:
                plt.plot(
                    group['chosen_layer'],
                    group[categories[i]],
                    marker='o',
                    label=f'ESM-2 model {model_name}'
                )

        # Set labels and title
        plt.legend(fontsize=20)
        plt.xlabel('Chosen Layer', fontsize=20)
        plt.ylabel(ylabels[i], fontsize=20)
        plt.title(titles[i], fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        # Show the plot
        plt.grid(True)
        plt.show()
