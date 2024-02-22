import numpy as np
import pandas as pd
from pynndescent import NNDescent
import esm
import torch
import pickle
from sklearn import metrics
import matplotlib.pyplot as plt
import scipy
import seaborn as sns

def get_esm_embedding(batch_converter, model, alphabet, data):

    sequence_representations = []

    # Iterate through the list in bulks because the execution is more stable that way.
    bulk_size = 500
    for i in range(0, len(data), bulk_size):
        print("i:", i)
        bulk = data[i:i + bulk_size]

        # Use batch converter as done in ESM-2 example.
        batch_labels, batch_strs, batch_tokens = batch_converter(bulk)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        # Extract per-residue representations (on CPU) as done in ESM-2 example.
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[6], return_contacts=True)
        token_representations = results["representations"][6]

        # Generate per-sequence representations via averaging as done in ESM-2 example.
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))

    return sequence_representations

# load lists
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

# prepare ESM
# models: esm2_t6_8M_UR50D, esm2_t33_650M_UR50D, esm2_t36_3B_UR50D (6GB), esm2_t48_15B_UR50D (30GB)
model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

# compute ESM embedding
false_neighbour_embeddings = get_esm_embedding(batch_converter, model, alphabet, false_neighbour_list)
true_neighbour_embeddings = get_esm_embedding(batch_converter, model, alphabet, true_neighbour_list)

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
ax.set_title('False transfer | Distribution of ESM distances with sklearn')
ax.set_xlabel('Distance')
ax.set_ylabel('Number of distances')
ax.grid(True)
ax.set_axisbelow(True)
plt.show()

fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.hist(false_transfer_distance_df_pynn, bins=40, color='blue', edgecolor='black', linewidth=1.2)
ax2.set_title('False transfer | Distribution of ESM distances with PyNNDescent')
ax2.set_xlabel('Distance')
ax2.set_ylabel('Number of distances')
ax2.grid(True)
ax2.set_axisbelow(True)
plt.show()

fig, ax = plt.subplots(figsize=(6, 6))
ax.hist(true_transfer_distance_df_sklearn, bins=40, color='blue', edgecolor='black', linewidth=1.2)
ax.set_title('True transfer | Distribution of ESM distances with sklearn')
ax.set_xlabel('Distance')
ax.set_ylabel('Number of distances')
ax.grid(True)
ax.set_axisbelow(True)
plt.show()

fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.hist(true_transfer_distance_df_pynn, bins=40, color='blue', edgecolor='black', linewidth=1.2)
ax2.set_title('True transfer | Distribution of ESM distances with PyNNDescent')
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
true_transfer_distance_df_pynn_cosine = true_transfer_distance_df_pynn_cosine.head(min_length)
false_transfer_distance_df_pynn_cosine = false_transfer_distance_df_pynn_cosine.head(min_length)

sklearn_ttest_result = scipy.stats.ttest_ind(true_transfer_distance_df_sklearn, false_transfer_distance_df_sklearn)
pynn_ttest_result = scipy.stats.ttest_ind(true_transfer_distance_df_pynn, false_transfer_distance_df_pynn)
pynn_ttest_result_cosine = scipy.stats.ttest_ind(true_transfer_distance_df_pynn_cosine, false_transfer_distance_df_pynn_cosine)
sklearn_wilcoxon_result = scipy.stats.wilcoxon(true_transfer_distance_df_sklearn, false_transfer_distance_df_sklearn)
pynn_wilcoxon_result = scipy.stats.wilcoxon(true_transfer_distance_df_pynn, false_transfer_distance_df_pynn)
pynn_wilcoxon_result_cosine = scipy.stats.wilcoxon(true_transfer_distance_df_pynn_cosine, false_transfer_distance_df_pynn_cosine)
#sklearn_wasserstein_result = scipy.stats.wasserstein_distance(true_transfer_distance_df_sklearn, false_transfer_distance_df_sklearn)
#pynn_wasserstein_result = scipy.stats.wasserstein_distance(true_transfer_distance_df_pynn, false_transfer_distance_df_pynn)
print('sklearn ttest result: ', sklearn_ttest_result.pvalue)
print('pynn cosine ttest result: ', pynn_ttest_result_cosine.pvalue)
print('pynn ttest result: ', pynn_ttest_result.pvalue)
print('sklearn wilcoxon result: ', sklearn_wilcoxon_result.pvalue)
print('pynn cosine wilcoxon result: ', pynn_wilcoxon_result_cosine.pvalue)
print('pynn wilcoxon result: ', pynn_wilcoxon_result.pvalue)
#print('sklearn wasserstein result: ', sklearn_wasserstein_result)
#print('pynn wasserstein result: ', pynn_wasserstein_result)

sns.set(rc={'figure.figsize':(14,14)})
sns.set_style("whitegrid", {'axes.grid' : False})

df = pd.DataFrame()
df['ESM (medium)\ntrue transfer\nSklearn distances'] = true_transfer_distance_df_sklearn
df['ESM (medium)\nfalse transfer\nSklearn distances'] = false_transfer_distance_df_sklearn
df['ESM (medium)\ntrue transfer\nPyNNDescent (euclidean)\ndistances'] = true_transfer_distance_df_pynn
df['ESM (medium)\nfalse transfer\nPyNNDescent (euclidean)\ndistances'] = false_transfer_distance_df_pynn
df['ESM (medium)\ntrue transfer\nPyNNDescent (cosine)\ndistances'] = true_transfer_distance_df_pynn_cosine
df['ESM (medium)\nfalse transfer\nPyNNDescent (cosine)\ndistances'] = false_transfer_distance_df_pynn_cosine

ax = sns.boxplot(data=df, palette = 'Reds', width = 0.5)
plt.show()