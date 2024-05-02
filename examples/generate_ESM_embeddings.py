import numpy as np
import esm
import torch

'''
This script is used to generate ESM-2 embeddings for different models, configurations and datasets.
It expects the file example_data/all_prototypes.npz (contains 86,705 prototypes) for the HeLa_Yeast_Ecoli example
and the file example_data/all_prototypes_H.npz (contains around 65,000 prototypes) for the HeLa-only example.
On my machine, the computation with the smallest model "esm2_t6_8M_UR50D" took around 10 minutes for only the 
example_data/all_prototypes.npz dataset. With the medium sized model "esm2_t33_650M_UR50D" it took around 10 hours.
If you want to generate embeddings with multiple different (large model) configurations, this code should be
parallelized (e.g. instead of using a for loop).
'''

def get_esm_embedding(batch_converter, model, alphabet, data, layer_number):

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
            results = model(batch_tokens, repr_layers=[layer_number], return_contacts=True)
        token_representations = results["representations"][layer_number]

        # Generate per-sequence representations via averaging as done in ESM-2 example.
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))

    return sequence_representations


'''
Here you can specify a list of ESM-2 model names for which to produce embeddings.
The following ESM-2 models are available and are downloaded in the first use: 
    - "esm2_t6_8M_UR50D": 6 layers, 8 million parameters, 320-dim embedding
    - "esm2_t12_35M_UR50D": 12 layers, 35 million parameters, 480-dim embedding
    - "esm2_t30_150M_UR50D": 30 layers, 150 million parameters, 640-dim embedding
    - "esm2_t33_650M_UR50D": 33 layers, 650 million parameters, 1280-dim embedding
    - "esm2_t36_3B_UR50D": : 36 layers, 3 billion parameters, 2560-dim embedding, 6GB size
    - "esm2_t48_15B_UR50D": 48 layers, 15 billion parameters, 5120-dim embedding, 30GB size
In the model names, the number after the "t" stands for the total layer number.
The next number stands for the parameter number.
"UR50D" stands for the used training dataset of the ESM-2 model which is UniRef 50.
'''
model_names = ['esm2_t6_8M_UR50D']

'''
This specifies the ESM-2 layer from which to retrieve the embeddings. Lower layers between 4 and 6 achieved the
best results. 
'''
representation_layers = [6]
datasets = ['HeLa', 'HeLa_Yeast_Ecoli']

for model_name in model_names:

    for layer in representation_layers:
        print("Representation layer: ", layer)
        for dataset in datasets:
            if dataset == 'HeLa':
                prototypes = np.load('example_data/all_prototypes_H.npz')
            elif dataset == 'HeLa_Yeast_Ecoli':
                prototypes = np.load('example_data/all_prototypes.npz')
            prototypes = dict(prototypes)
            model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
            batch_converter = alphabet.get_batch_converter()
            model.eval()  # disables dropout for deterministic results
            esm_input = list(zip(prototypes['PrecursorID'], prototypes['Sequence']))
            print("Start computing embeddings...")
            esm_embeddings = get_esm_embedding(batch_converter, model, alphabet, esm_input, layer)
            esm_embeddings_np = np.array(esm_embeddings)
            np.save('example_data/embeddings_{}_layer{}_dataset{}'.format(model_name, layer, dataset),
                    esm_embeddings_np)
            print(f"Finished computing embeddings for model '{model_name}', layer {layer} and dataset '{dataset}'.")
