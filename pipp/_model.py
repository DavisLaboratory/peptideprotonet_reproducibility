import logging

from typing import Tuple, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from pynndescent import NNDescent
from scipy.special import softmax

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
        self,
        ms: pd.DataFrame,
        msms: pd.DataFrame,
        k_neighbours=5,
        use_anchors=True,
        verbose=True,
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

        prototypes = self._compute_prototypes(msms, verbose=verbose)

        identities, confidence = self._propagate_using_prototypes(
            ms,
            prototypes,
            k_neighbours=k_neighbours,
            use_anchors=use_anchors,
            verbose=verbose,
        )

        return identities, confidence

    def _propagate_using_prototypes(
        self,
        ms: pd.DataFrame,
        prototypes: Dict[str, np.ndarray],
        k_neighbours=5,
        distance_metric="euclidean",
        use_anchors=True,
        verbose=True,
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

        distance_metric
            Distance metric to use when computing neighbours.

        use_anchors
            Whether to compute a relative representation.

        verbose
            Whether to print progress.

        Returns
        -------
            Predicted identities and confidence.
        """

        query_embeddings = self.get_latent_representations(ms[self._features])
        query_charges = ms["Charge"].values

        prototype_embeddings = prototypes["Embedding"]
        prototype_charges = prototypes["Charge"]

        if use_anchors:
            # normalize embeddings
            means = np.mean(query_embeddings, axis=0)
            query_embeddings -= means
            prototype_embeddings -= means

            anchors = self._select_anchors(prototype_embeddings)

            prototype_representation = self._compute_relative_representations(
                prototype_embeddings, anchors
            )
            query_representation = self._compute_relative_representations(
                query_embeddings, anchors
            )

            # recover original embeddings - keep in-place to save memory
            query_embeddings += means
            prototype_embeddings += means

        else:
            prototype_representation = prototype_embeddings
            query_representation = query_embeddings

        knn_index = NNDescent(
            prototype_representation, metric=distance_metric, n_jobs=-1
        )

        if verbose:
            print(f"computing {k_neighbours}-nearest neighbour prototypes...")

        neighbours, distances = knn_index.query(query_representation, k=k_neighbours)

        neighbours_weights = self._compute_weights(distances)
        neighbours_charges = np.array(
            [prototype_charges[q_neighbours] for q_neighbours in neighbours]
        )

        if verbose:
            print("computing confidence and selecting identities...")

        identities_args, confidence = self._compute_prediction_with_charge_filter(
            query_charges, neighbours_weights, neighbours_charges
        )

        identities_ids = neighbours[
            np.arange(identities_args.shape[0]), identities_args
        ]
        identities = prototypes["PrecursorID"][identities_ids]

        return identities, confidence

    def _select_anchors(
        self, latent_embeddings: np.ndarray, n_anchors=80
    ) -> np.ndarray:
        """
        Select anchors from the latent embeddings using uniform sampling.

        Parameters
        ----------
        latent_embeddings
            Latent embeddings to consider when selecting anchors.

        n_anchors
            Number of anchors to select.

        Returns
        -------
            Anchors.
        """

        anchors_idx = np.random.choice(
            latent_embeddings.shape[0], size=n_anchors, replace=False
        )
        anchors = latent_embeddings[anchors_idx]

        return anchors.copy()

    def _compute_relative_representations(
        self, xs: np.ndarray, anchors: np.ndarray
    ) -> np.ndarray:
        """
        Helper function to compute relative representations.

        Parameters
        ----------
        xs
            Embeddings.

        anchors
            Anchors.

        Returns
        -------
            Relative representations.
        """

        assert (
            xs.shape[1] == anchors.shape[1]
        ), "xs and anchors must have the same number of features"

        # use cosine-simularity to compute relative representations
        anchors_norm = np.linalg.norm(anchors, axis=1)
        xs_norm = np.linalg.norm(xs, axis=1)
        cosine_simularity = np.dot(anchors, xs.T) / np.outer(anchors_norm, xs_norm)
        relative_representation = cosine_simularity.T

        return relative_representation

    def _compute_prototypes(
        self, msms: pd.DataFrame, verbose=False
    ) -> Dict[str, np.ndarray]:
        """
        Helper function to compute prototypes.

        Parameters
        ----------
        msms
            Support set with the columns: ['PrecursorID', 'Charge', 'Mass', 'm/z', 'Retention time', 'Retention length', 'Ion mobility index', 'Ion mobility length', 'Number of isotopic peaks']

        verbose
            Whether to print progress.

        Returns
        -------
            Prototypes with the columns: ['PrecursorID', 'Charge', 'Embedding']
        """

        if verbose:
            print("computing prototypes...")

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
        stds = stds**2
        stds = stds.reshape(-1, 1)
        distances_tilda = np.exp(-np.true_divide(distances**2, stds))
        return distances_tilda

        """
        # convert distances to affinities
        stds = np.mean(distances, axis=1)
        stds = (stds) ** 2
        stds = stds.reshape(-1, 1)
        distances_tilda = np.exp(-np.true_divide(distances ** 2, stds))
        return distances_tilda
#         stds = np.std(distances, axis=1)
#         stds = (2.0 / stds) ** 2
#         stds = stds.reshape(-1, 1)
#         distances_tilda = np.exp(-np.true_divide(distances, stds))
#         return distances_tilda
#         return 1.0-distances # cosine similarity


        # @NOTE: handle division-by-0, by setting the output "weight" to 0 instead of nan.
        # weights = distances_tilda / np.sum(distances_tilda, axis=1, keepdims=True)
#         weights = np.divide(
#             distances_tilda,
#             np.sum(distances_tilda, axis=1, keepdims=True),
#             out=np.zeros_like(distances_tilda),
#             where=distances_tilda != 0,
#         )
#         weights = softmax(distances, axis=1)

#         return weights


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

        filtered_weights = neighbours_weights * (
            neighbours_charges == query_charges.reshape(-1, 1)
        )

        denominator = np.sum(filtered_weights, axis=1).reshape(-1, 1)
        denominator[denominator == 0] = 1


        # normalize confidences based on charges between neighbour and query matching
        probs = filtered_weights / denominator

        predictions = np.argmax(probs, axis=1)
        confidence = np.max(probs, axis=1)


        return predictions, confidence
