#!/usr/bin/env python3

from __future__ import print_function

import argparse
import numpy as np
import pickle


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, random_split

import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels

from sklearn.preprocessing import StandardScaler







def index_classes(items):
    idx = {}
    for i in items:
        if (i not in idx):
            idx[i] = len(idx)
    return idx


class CreatePeakApexDatasetFromEvidence(Dataset):
    """
    [[Source]](https://github.com/repo/blob/master/protonet_peakapexMQ.py)
    **Description**
    Creates a dataset of peptide precursor peak apex attributes required for peptideprotonet from evidence table by MaxQuant
    **Arguments**
    * **pickle_file** (str) - Path to pickle_file containing evidence table(s).
    **Example**
    ~~~python
    train_dataset = CreatePeakApexDatasetFromPickle(pickle_file = "path/to/evidence.pkl")
    ~~~
    """

    def __init__(self, pickle_file):
        super(CreatePeakApexDatasetFromEvidence, self).__init__()
        
        with open(pickle_file, 'rb') as f:
            self.data = pickle.load(f)

                

                
        attr_names = ['Charge','Mass', 'm/z', 'Retention time','Retention length', 'Ion mobility index', 
        'Ion mobility length','Number of isotopic peaks'] 
        
        
        features = attr_names
        
        self.data.reset_index(drop=True, inplace=True)
        self.data['PrecursorID'] = self.data['Modified sequence'].astype(str).str.cat(self.data.Charge.astype(str), sep='_')
        
        self.x = torch.from_numpy(StandardScaler().fit_transform(self.data[features].to_numpy())).float()
        
        
        c = self.data['study'].to_numpy() # assumes column exits
        conditions = np.unique(c)

        condition_encoder = {k: v for k, v in zip(conditions, range(len(conditions)))}

        c_labels = np.zeros(c.shape[0])
        for condition, label in condition_encoder.items():
            c_labels[c == condition] = label
        c = torch.from_numpy(c_labels.astype('int'))
        c = c.view(-1,1)
        
        self.x = torch.cat((self.x, c), dim = 1)
        self.y = np.ones(len(self.x))
        
        
        
        class_dict = {}
        for label in set(self.data.PrecursorID):
            idxs = self.data.loc[self.data.PrecursorID == label].index.tolist()
            class_dict[label] = idxs



        # TODO Remove index_classes from here
        self.class_idx = index_classes(class_dict.keys())
        for class_name, idxs in class_dict.items():
            for idx in idxs:
                self.y[idx] = self.class_idx[class_name]

    def __getitem__(self, idx):
        data = self.x[idx]
        return data, self.y[idx]

    def __len__(self):
        return len(self.x)







def pairwise_distances_logits(a, b):
    n = a.shape[0]
    m = b.shape[0]
    logits = -((a.unsqueeze(1).expand(n, m, -1) -
                b.unsqueeze(0).expand(n, m, -1))**2).sum(dim=2)
    return logits


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


class Encoder(nn.Module):

    def __init__(self, input_dim=8, hidden_dim=64, latent_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU()
            )

        self.mean_encoder = nn.Linear(hidden_dim, latent_dim)
        self.var_encoder = nn.Linear(hidden_dim, latent_dim)


    def forward(self, x):
        # Simple forward
        hidden = self.encoder(x)
        mu = self.mean_encoder(hidden)
        logvar = self.var_encoder(hidden)
        std = logvar
        eps = torch.randn_like(std)
        x_sample = mu
        return x_sample
        


def fast_adapt(model, batch, ways, shot, query_num, metric=None, device=None):
    if metric is None:
        metric = pairwise_distances_logits
    if device is None:
        device = model.device()
    data, labels = batch
    data = data.to(device)
    labels = labels.to(device)
    n_items = shot * ways

    # Sort data samples by labels
    # TODO: Can this be replaced by ConsecutiveLabels ?
    sort = torch.sort(labels)
    data = data.squeeze(0)[sort.indices].squeeze(0)
    labels = labels.squeeze(0)[sort.indices].squeeze(0)


    c = data[:,-1]
    c = F.one_hot(torch.from_numpy(c.detach().cpu().numpy().astype('int')))
    c = c.to(device)
    data = data[:,:-1] # firt dimension is batch size, 2nd is number of samples (query+ support), 3rd is number of features

    # Compute support and query embeddings
    embeddings = model(data)
    support_indices = np.zeros(data.size(0), dtype=bool)
    selection = np.arange(ways) * (shot + query_num)
    for offset in range(shot):
        support_indices[selection + offset] = True
    query_indices = torch.from_numpy(~support_indices)
    support_indices = torch.from_numpy(support_indices)
    support = embeddings[support_indices]

    support = torch.cat((support, c[support_indices]), dim=1)

    support = support.reshape(ways, shot, -1).mean(dim=1)
    
    query = embeddings[query_indices]
    
    query = torch.cat((query,c[query_indices]), dim=1)


    labels = labels[query_indices].long()
    logits = pairwise_distances_logits(query, support)
    loss = F.cross_entropy(logits, labels)
    acc = accuracy(logits, labels)
    return loss, acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=250)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--test-shot', type=int, default=1)
    parser.add_argument('--test-query', type=int, default=1)
    parser.add_argument('--train-query', type=int, default=1)
    parser.add_argument('--train-way', type=int, default=5)
    parser.add_argument('--gpu', default=0)
    args = parser.parse_args()
    print(args)

    device = torch.device('cpu')
    if args.gpu and torch.cuda.device_count():
        print("Using gpu")
        torch.cuda.manual_seed(43)
        device = torch.device('cuda')

    model = Encoder()
    model.to(device)


    print('preparing training and validation data')
    path_data = '../ion_mobility/PXD019086_PXD010012_combined_evidence_train_90Kto20Ksplit_5query_1shot_allPeptidesTxtFeatures_modSeqSpecies.pkl'
    train_dataset = CreatePeakApexDatasetFromEvidence(path_data)
    train_dataset = l2l.data.MetaDataset(train_dataset)
    train_transforms = [
        NWays(train_dataset, args.train_way),
        KShots(train_dataset, args.train_query + args.shot),
        LoadData(train_dataset),
        RemapLabels(train_dataset),
    ]
    train_tasks = l2l.data.TaskDataset(train_dataset, task_transforms=train_transforms)
    train_loader = DataLoader(train_tasks, pin_memory=True, shuffle=True)

    
    path_valid_data = '../ion_mobility/PXD019086_PXD010012_combined_evidence_valid_90Kto20Ksplit_5query_1shot_allPeptidesTxtFeatures_modSeqSpecies.pkl'
    valid_dataset = CreatePeakApexDatasetFromEvidence(path_valid_data)

    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    valid_transforms = [
        NWays(valid_dataset, args.test_way),
        KShots(valid_dataset, args.test_query + args.test_shot),
        LoadData(valid_dataset),
        RemapLabels(valid_dataset),
    ]
    valid_tasks = l2l.data.TaskDataset(valid_dataset,
                                       task_transforms=valid_transforms,
                                       num_tasks=5000)
    valid_loader = DataLoader(valid_tasks, pin_memory=True, shuffle=True)

 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.5)

    print('Training')
    for epoch in range(1, args.max_epoch + 1):
        model.train()

        loss_ctr = 0.01
        n_loss = 0
        n_acc = 0

        for i in range(100):
            try:
                batch = next(iter(train_loader))
                #print(batch)
            except ValueError:
                print('skipped training iteration')
                continue

            loss, acc = fast_adapt(model,
                                   batch,
                                   args.train_way,
                                   args.shot,
                                   args.train_query,
                                   metric=pairwise_distances_logits,
                                   device=device)

            loss_ctr += 1
            n_loss += loss.item()
            n_acc += acc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_scheduler.step()

        print('epoch {}, train, loss={:.4f} acc={:.4f}'.format(
            epoch, n_loss/loss_ctr, n_acc/loss_ctr))

        model.eval()

        loss_ctr = 0
        n_loss = 0
        n_acc = 0
        for i  in range(len(valid_loader)):
            try:
                batch = next(iter(valid_loader))
                #print(batch)
            except ValueError:
                print('skipped validation iteration')
                continue
            loss, acc = fast_adapt(model,
                                   batch,
                                   args.test_way,
                                   args.test_shot,
                                   args.test_query,
                                   metric=pairwise_distances_logits,
                                   device=device)


            


            loss_ctr += 1
            n_loss += loss.item()
            n_acc += acc

        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(
            epoch, n_loss/loss_ctr, n_acc/loss_ctr))


    torch.save(model, 'PXD019086_PXD010012_combined_evidence_90Kto20Ksplit_5query_1shot_fullmodel_featuresScaled_allPeptidesTxtFeatures_modSeqSpecies_hidden64_latent10_maxEpoch400_164trainways_2ndlayer_conditionalEmbedding.pth')
    print('model saved')
