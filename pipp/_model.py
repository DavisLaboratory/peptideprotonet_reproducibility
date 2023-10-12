import logging

from typing import Tuple, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from pynndescent import NNDescent

from ._module import Encoder

logger = logging.getLogger(__name__)


class Peptideprotonet:
    def __init__(self, device: str = None, dir_path: str = None):
        super().__init__()

        self._features = [
            "Charge",
            "Mass",
            "m/z",
            "Retention time",
            "Retention length",
            "Ion mobility index",
            "Ion mobility length",
            "Number of isotopic peaks",
        ]

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.module = Encoder()

        if dir_path:
            self.module.load_state_dict(
                torch.load(dir_path, map_location=torch.device("cpu"))
            )

        self.module.to(device)
        self.module.eval()

    @classmethod
    def load(clz, dir_path: str, device: str = None) -> "Peptideprotonet":
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

    def get_latent_representations(self, x: pd.DataFrame) -> np.ndarray:
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

        features = [
            "Charge",
            "Mass",
            "m/z",
            "Retention time",
            "Retention length",
            "Ion mobility index",
            "Ion mobility length",
            "Number of isotopic peaks",
        ]

        x = x[features]

        x = StandardScaler().fit_transform(x.to_numpy())
        x = torch.from_numpy(x).float().to(self.device)
        z = self.module(x)

        latent = z.cpu().detach().numpy()

        return latent

    def propagate(
        self, ms: pd.DataFrame, msms: pd.DataFrame, k_neighbours=5, verbose=True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate the identities/labels from the support set to the query set.

        Parameters
        ----------
        ms
            Query set with the columns: ['Charge', 'Mass', 'm/z', 'Retention time', 'Retention length', 'Ion mobility index', 'Ion mobility length', 'Number of isotopic peaks']

        msms
            Support set with the columns: ['PrecursorID', 'Charge', 'Mass', 'm/z', 'Retention time', 'Retention length', 'Ion mobility index', 'Ion mobility length', 'Number of isotopic peaks']

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
        >>> ms = pd.read_csv('path/to/ms.csv')
        >>> msms = pd.read_csv('path/to/msms.csv')
        >>> identities, confidence = model.propagate(ms, msms)
        """

        if verbose:
            print("Computing prototypes...")

        prototypes = self._compute_prototypes(msms)

        if verbose:
            print(f"Propagating identities using {k_neighbours} nearest neighbours...")

        identities, confidence = self._propagate_using_prototypes(
            ms, prototypes, k_neighbours=k_neighbours
        )

        return identities, confidence

    def _propagate_using_prototypes(
        self, ms: pd.DataFrame, prototypes: Dict[str, np.ndarray], k_neighbours=5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Helper function to propagate identities using prototypes.

        Parameters
        ----------
        MS
            Query set with the columns: ['charge', 'mass', 'm/z', 'retention_time', 'retention_length', 'ion_mobility_index', 'ion_mobility_length', 'num_isotopic_peaks']

        prototypes
            Prototypes with the columns: ['precursor_id', 'charge', 'embedding']

        k_neighbours
            Number of neighbours to consider when computing identities and confidence.

        Returns
        -------
            Predicted identities and confidence.
        """

        query_embeddings = self.get_latent_representations(ms[self._features])
        query = {"Charges": ms["Charge"].values, "Embedding": query_embeddings}

        knn_index = NNDescent(prototypes["Embedding"], metric="cosine", n_jobs=-1)
        neighbours, distances = knn_index.query(query["Embedding"], k=k_neighbours)

        neighbours_weights = self._compute_weights(distances)
        neighbours_charges = np.array(
            [prototypes["Charge"][q_neighbours] for q_neighbours in neighbours]
        )

        identities_args, confidence = self._compute_prediction_with_charge_filter(
            query["Charges"], neighbours_weights, neighbours_charges
        )

        identities_ids = neighbours[
            np.arange(identities_args.shape[0]), identities_args
        ]
        identities = prototypes["PrecursorID"][identities_ids]

        return identities, confidence

    def _compute_prototypes(self, msms: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Helper function to compute prototypes.

        Parameters
        ----------
        msms
            Support set with the columns: ['PrecursorID', 'Charge', 'Mass', 'm/z', 'Retention time', 'Retention length', 'Ion mobility index', 'Ion mobility length', 'Number of isotopic peaks']

        Returns
        -------
            Prototypes with the columns: ['PrecursorID', 'Charge', 'Embedding']
        """

        support_embeddings = self.get_latent_representations(msms[self._features])

        prototypes = []  # list of (precursor_id, charge, embedding)
        precursor_groups = msms.groupby(["PrecursorID"])

        for group in precursor_groups:
            (precursor_id,) = group[0]
            locs = group[1].index
            charge = msms.loc[locs[0], "Charge"]
            ilocs = msms.index.get_indexer(locs)
            prototypes.append(
                (precursor_id, charge, np.mean(support_embeddings[ilocs], axis=0))
            )

        precursor_ids, charges, embeddings = zip(*prototypes)

        precursor_ids = np.array(precursor_ids)
        charges = np.array(charges)
        embeddings = np.array(embeddings)

        prototypes = {
            "PrecursorID": precursor_ids,
            "Charge": charges,
            "Embedding": embeddings,
        }

        return prototypes

    def _compute_weights(self, distances: np.ndarray) -> np.ndarray:
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

        stds = np.mean(distances, axis=1)
        stds = (stds) ** 2
        stds = stds.reshape(-1, 1)
        distances_tilda = np.exp(-np.true_divide(distances ** 2, stds))
        return distances_tilda

        """
        # convert distances to affinities
        stds = np.std(distances, axis=1)
        stds = (2.0 / stds) ** 2
        stds = stds.reshape(-1, 1)
        distances_tilda = np.exp(-np.true_divide(distances, stds))

        # @NOTE: handle division-by-0, by setting the output "weight" to 0 instead of nan.
        # weights = distances_tilda / np.sum(distances_tilda, axis=1, keepdims=True)
        weights = np.divide(
            distances_tilda,
            np.sum(distances_tilda, axis=1, keepdims=True),
            out=np.zeros_like(distances_tilda),
            where=distances_tilda != 0,
        )

        return weights
        """

    def _compute_prediction_with_charge_filter(
        self,
        query_charges: np.ndarray,
        neighbours_weights: np.ndarray,
        neighbours_charges: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        predictions = np.zeros((N,), dtype=np.int32)
        confidence = np.zeros((N,), dtype=np.float32)

        for i in range(N):
            query_weights = neighbours_weights[i]
            query_charge = query_charges[i]
            nb_charges = neighbours_charges[i]

            ps = query_weights * (query_charge == nb_charges)

            # @NOTE: handle division-by-0, by setting the output "weight" to 0 instead of nan.
            # probs = ps / np.sum(ps)
            probs = np.divide(ps, np.sum(ps), out=np.zeros_like(ps), where=ps != 0)

            predictions[i] = np.argmax(probs)
            confidence[i] = np.max(probs)

        return predictions, confidence
