from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
import numpy as np
import pandas as pd
import random
import umap
import matplotlib.pyplot as plt

'''
This script is used for the following tasks:

1. Generate and store ProtT5 embeddings with the function call_protT5().
   This function expects the file "example_data/all_prototypes.npz" (contains 87605 prototypes).
   This function stores ProtT5 embeddings at 'example_data/protT5_embeddings_df.pkl'.
2. Plot precomputed ProtT5 embeddings with the function plot_stored_embeddings().
   This function expects the file 'example_data/protT5_embeddings_df.pkl' 
   
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
