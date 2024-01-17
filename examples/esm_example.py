import random

import torch
import esm
import umap
import matplotlib.pyplot as plt

# create function with model variable and input variable

# Load ESM-2 model
model, alphabet = (esm.pretrained.esm2_t33_650M_UR50D())
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

# Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
# use protein names or leading razor protein as identifier/tag
# use unmodified sequence
data = [
    ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
    ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    ("protein2 with mask", "KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    ("protein3",  "K A <mask> I S Q"),
    ("protein4", "GLGAAEFGGAAGNVEAPGETFAQR"),
]

#data = data * 500
#random_subset_with_duplicates = random.choices(data, k=1000)

batch_labels, batch_strs, batch_tokens = batch_converter(data)
batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

# Extract per-residue representations (on CPU)
# look at different layer numbers
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=False)
token_representations = results["representations"][33]

# Generate per-sequence representations via averaging
# NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
sequence_representations = []
for i, tokens_len in enumerate(batch_lens):
    sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

# Look at the unsupervised self-attention map contact predictions
# for (_, seq), tokens_len, attention_contacts in zip(data, batch_lens, results["contacts"]):
#    plt.matshow(attention_contacts[: tokens_len, : tokens_len])
#    plt.title(seq)
#    plt.show()

# reduce representation with umap scanpy, umap documentation
reducer = umap.UMAP(metric = 'cosine')
umap_embedding = reducer.fit_transform(sequence_representations)
fig, ax = plt.subplots(figsize=(14, 12), ncols=1, nrows=1)

sp = ax.scatter(umap_embedding[:,0], umap_embedding[:,1], s=0.1)
ax.set_xlabel('UMAP1')
ax.set_ylabel('UMAP2')
ax.set_title('PLM embedding')
fig.colorbar(sp)
plt.show()
# colored by species
# compare ESM dimesnions (small , medium, big)
# goal: get representation that seperates species
# metric for comparing species spearation? eg in comparison with protT5