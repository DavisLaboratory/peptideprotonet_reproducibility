import logging

from typing import Tuple

import numpy as np
import pandas as pd
import torch

from ._module import Encoder

from sklearn.preprocessing import StandardScaler
from pynndescent import NNDescent

logger = logging.getLogger(__name__)


class Peptideprotonet:

    def __init__(self, device:str=None, dir_path:str=None):
        super().__init__()
        
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        self.device = torch.device(device)

        self.module = Encoder()

        if dir_path:
            self.module.load_state_dict(torch.load(dir_path, map_location=torch.device('cpu')))

        self.module.to(device)
        self.module.eval()


    # @NOTE: this loading method will be inherited later
    @classmethod
    def load(clz, dir_path:str, device:str=None) -> 'Peptideprotonet':
        """
        Instantiate a Peptideprotonet model from a pretrained model.

        Parameters
        ----------

        dir_path
            Path to pretrained model.
        
        device
            Device to load the model on.

        Returns
        -------
            Model with pretrained weights.

        Example
        -------
        >>> import pipp
        >>> model = pipp.Peptideprotonet.load('path/to/model.pt')
        """

        return Peptideprotonet(device=device, dir_path=dir_path)

    def get_latent_representations(self, x:pd.DataFrame) -> np.ndarray:
        """
        Get the latent representation of the data.

        Parameters
        ----------
        x
            Dataframe with the columns: ['Charge','Mass', 'm/z', 'Retention time', 'Retention length', 'Ion mobility index', 'Ion mobility length', 'Number of isotopic peaks']

        Returns
        -------
            Embeddings of the data.
        """
        
        features = ['Charge','Mass', 'm/z', 'Retention time', 'Retention length', 'Ion mobility index', 'Ion mobility length', 'Number of isotopic peaks']
        x = x[features]
        
        x = StandardScaler().fit_transform(x.to_numpy())
        x = torch.from_numpy(x).float().to(self.device)
        z = self.module(x)
        
        latent = z.cpu().detach().numpy()

        return latent
    

    def propagate(self, MS:pd.DataFrame, MSMS:pd.DataFrame, k_neighbours=5, verbose=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate the identities/labels from the support set to the query set.

        Parameters
        ----------
        MS
            Query set with the columns: ['Charge','Mass', 'm/z', 'Retention time', 'Retention length', 'Ion mobility index', 'Ion mobility length', 'Number of isotopic peaks']
        
        MSMS
            Support set with the columns: ['PrecursorID', 'Charge','Mass', 'm/z', 'Retention time', 'Retention length', 'Ion mobility index', 'Ion mobility length', 'Number of isotopic peaks']
        
        k_neighbours
            Number of neighbours to consider when computing identities and confidence.

        verbose
            Whether to print progress.

        Returns
        -------
            Predicted identities and confidence.

        Example
        -------
        >>> import pipp
        >>> model = pipp.Peptideprotonet.load('path/to/model.pt')
        >>> MS = pd.read_csv('path/to/MS.csv')
        >>> MSMS = pd.read_csv('path/to/MSMS.csv')
        >>> identities, confidence = model.propagate(MS, MSMS)
        """

        features = ['Charge','Mass', 'm/z', 'Retention time', 'Retention length', 'Ion mobility index', 'Ion mobility length', 'Number of isotopic peaks']

        # compute peptide embeddings
        if verbose:
            print("Computing peptide embeddings...")

        query_embeddings = self.get_latent_representations(MS[features])
        support_embeddings = self.get_latent_representations(MSMS[features])

        # compute the prototype embeddings
        if verbose:
            print("Computing prototype embeddings...")

        prototypes = [] # list of (precursor_id, charge, embedding)
        precursor_groups = MSMS.groupby(['PrecursorID'])

        for group in precursor_groups:
            precursor_id, = group[0]
            locs = group[1].index
            charge = MSMS.loc[locs[0], 'Charge']
            ilocs = MSMS.index.get_indexer(locs)
            prototypes.append((precursor_id, charge, np.mean(support_embeddings[ilocs], axis=0)))

        prototype_precursor_ids, prototype_charges, prototype_embeddings = zip(*prototypes)

        prototype_precursor_ids = np.array(prototype_precursor_ids)
        prototype_charges = np.array(prototype_charges)
        prototype_embeddings = np.array(prototype_embeddings)

        query_charges = MS['Charge'].values

        if verbose:
            print("Propagating identities...")

        knn_index = NNDescent(prototype_embeddings, metric='euclidean', n_jobs=-1)
        neighbours, distances = knn_index.query(query_embeddings, k=k_neighbours)

        if verbose:
            print("Computing identities and confidence...")

        neighbours_weights = self._compute_weights(distances)
        neighbours_charges = np.array([prototype_charges[q_neighbours] for q_neighbours in neighbours])

        identities_args, confidence = self._compute_prediction_with_charge_filter(query_charges, neighbours_weights, neighbours_charges)

        identities_ids = neighbours[np.arange(identities_args.shape[0]), identities_args]
        identities = prototype_precursor_ids[identities_ids]

        return identities, confidence
    

    def _compute_weights(self, distances:np.ndarray) -> np.ndarray:
        """
            Helper function to compute weights from distances.

            Parameters
            ----------
            distances
                Distances between query and support set.
            
            Returns
            -------
                Weights for each neighbour.
        """
        
        # convert distances to affinities
        stds = np.std(distances, axis=1)
        stds = (2.0 / stds) ** 2
        stds = stds.reshape(-1, 1)
        distances_tilda = np.exp(-np.true_divide(distances, stds))

        # @NOTE: handle division-by-0, by setting the output "weight" to 0 instead of nan.
        #weights = distances_tilda / np.sum(distances_tilda, axis=1, keepdims=True)
        weights = np.divide(distances_tilda, np.sum(distances_tilda, axis=1, keepdims=True), out=np.zeros_like(distances_tilda), where=distances_tilda != 0)
        
        return weights


    def _compute_prediction_with_charge_filter(self, query_charges:np.ndarray, neighbours_weights:np.ndarray, neighbours_charges:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
            Helper function to compute predictions with charge filter.

            Parameters
            ----------
            query_charges
                Charges of the query set.

            neighbours_weights
                Weights of the neighbours.

            neighbours_charges
                Charges of the neighbours.

            Returns
            -------
                Predictions and confidence.
        """

        N = neighbours_weights.shape[0]
        predictions = np.zeros((N,), dtype=np.int64)
        confidence = np.zeros((N,), dtype=np.float64)

        for i in range(N):
            query_weights = neighbours_weights[i]
            query_charge = query_charges[i]
            nb_charges = neighbours_charges[i]

            ps = query_weights * (query_charge == nb_charges)

            # @NOTE: handle division-by-0, by setting the output "weight" to 0 instead of nan.
            #probs = ps / np.sum(ps)
            probs = np.divide(ps, np.sum(ps), out=np.zeros_like(ps), where=ps != 0)
            
            predictions[i] = np.argmax(probs)
            confidence[i] = np.max(probs)

        return predictions, confidence

