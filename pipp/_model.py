import logging

from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pynndescent import NNDescent
import esm
import torch
from tqdm import tqdm

from typing import Literal

from ._module import Encoder

logger = logging.getLogger(__name__)


class Peptideprotonet:
    def __init__(self, device: str | None = None, dir_path: str | None = None):
        super().__init__()

        self.device = _get_device(device)

        self.ms1_module = Encoder()

        if dir_path:
            self.ms1_module.load_state_dict(
                torch.load(dir_path, map_location=torch.device("cpu"))
            )

        self.ms1_module.to(device)
        self.ms1_module.eval()

    @classmethod
    def load(clz, dir_path: str, device: str | None = None) -> "Peptideprotonet":
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

    def compute_ms1_embeddings(self, ms1: pd.DataFrame) -> np.ndarray:
        """
        Compute the MS1 embeddings of the data.

        Parameters
        ----------
        ms1
            Dataframe with the columns: ['Charge','Mass', 'm/z', 'Retention time', 'Retention length', 'Ion mobility index', 'Ion mobility length', 'Number of isotopic peaks']

        Returns
        -------
            MS1 Embeddings of the data.
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

        ms1 = ms1[features]

        x = StandardScaler().fit_transform(ms1.to_numpy())
        x = torch.from_numpy(x).float().to(self.device)

        with torch.no_grad():
            z = self.ms1_module(x)

        latent = z.cpu().numpy()

        return latent

    def get_prototype_sequences(self, msms: pd.DataFrame) -> Dict[str, str]:
        """
        Retrieve the sequences of the prototypes. Used to compute the sequence embeddings.
        """
        precursor_groups = msms.groupby(["PrecursorID"])
        precursors = [df.iloc[0]["PrecursorID"] for _, df in precursor_groups]
        sequences = [df.iloc[0]["Sequence"] for _, df in precursor_groups]
        return dict(zip(precursors, sequences))

    def propagate(
        self,
        ms: pd.DataFrame,
        msms: pd.DataFrame,
        *,
        k_neighbours=5,
        representation: Literal["absolute", "relative"] = "relative",
        n_anchors=300,
        ms2_embeddings: Dict[str, np.ndarray] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate the identities/labels from the support set to the query set.

        Parameters
        ----------
        ms
            Query set with the columns: ['Charge', 'Mass', 'm/z', 'Retention time', 'Retention length', 'Ion mobility index', 'Ion mobility length', 'Number of isotopic peaks']

        msms
            Support set from which the prototypes are taken.
            With the columns: ['PrecursorID', 'Charge', 'Mass', 'm/z', 'Retention time', 'Retention length', 'Ion mobility index', 'Ion mobility length', 'Number of isotopic peaks']

        k_neighbours
            Number of neighbours to consider when computing identities and confidence score.

        representation
            Type of representation to use. Either 'absolute' or 'relative'.

        n_anchors
            Number of anchors that are the basis for the relative representations.

        ms2_embeddings
            Sequence embeddings for each prototypes as a dictionary with the precursor ID as the key and the embedding as the value.

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

        self._verify_ms_data(ms)
        self._verify_msms_data(msms)

        use_anchors = representation == "relative"

        if ms2_embeddings is not None:
            assert use_anchors, "MS2 embeddings require relative representations to be used. Set representation='relative'."

        query = self._compute_query(ms)
        prototypes = self._compute_prototypes(msms)

        if ms2_embeddings is not None:
            self._attach_ms2_embeddings(prototypes, ms2_embeddings)

        query_representation, prototype_representation = self._compute_representations(
            query,
            prototypes,
            use_anchors,
            n_anchors=n_anchors,
        )

        neighbour_indices, neighbour_distances = self._find_k_closest_prototypes(
            query_representation,
            prototype_representation,
            k=k_neighbours,
        )

        neighbour_affinities = self._compute_affinities(neighbour_distances)
        neighbour_probabilities = self._compute_assignment_probabilities(
            neighbour_affinities,
            prototypes["Charge"][neighbour_indices],
            query["Charge"],
        )

        prototype_indices, confidence_scores = self._assign_prototype_to_query(
            neighbour_probabilities,
            neighbour_indices,
        )

        precursor_ids = prototypes["PrecursorID"][prototype_indices]

        return precursor_ids, confidence_scores

    def _compute_prototypes(
        self,
        msms: pd.DataFrame,
    ) -> Dict[str, np.ndarray]:
        ms1_embeddings = self.compute_ms1_embeddings(msms)

        prototypes_list = []
        precursor_groups = msms.groupby(["PrecursorID"])

        for _, df_group in precursor_groups:
            precursor = df_group.iloc[0]

            precursor_id = precursor["PrecursorID"]
            charge = precursor["Charge"]
            protein = precursor["Proteins"]
            sequence = precursor["Sequence"]
            n_runs = len(df_group)

            ilocs = msms.index.get_indexer(df_group.index)
            ms1_embedding = np.mean(ms1_embeddings[ilocs], axis=0)

            prototype = (precursor_id, charge, protein, sequence, n_runs, ms1_embedding)
            prototypes_list.append(prototype)

        keys = [
            "PrecursorID",
            "Charge",
            "Proteins",
            "Sequence",
            "Run_Count",
            "ms1_embedding",
        ]
        values = [np.array(v) for v in zip(*prototypes_list)]
        prototypes = dict(zip(keys, values))

        return prototypes

    def _attach_ms2_embeddings(
        self,
        prototypes: pd.DataFrame,
        ms2_embeddings: Dict[str, np.ndarray],
    ):
        prototypes["ms2_embedding"] = [
            ms2_embeddings[precursor] for precursor in prototypes["PrecursorID"]
        ]

    def _compute_query(self, ms: pd.DataFrame) -> Dict[str, np.ndarray]:
        query_ms1_embeddings = self.compute_ms1_embeddings(ms)
        query = {
            "Charge": ms["Charge"].values,
            "ms1_embedding": query_ms1_embeddings,
        }
        return query

    def _compute_representations(
        self,
        query: Dict[str, np.ndarray],
        prototypes: Dict[str, np.ndarray],
        use_anchors: bool,
        n_anchors=50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute representations for query and prototypes.

        Parameters
        ----------
        query
            Query set with the columns: ['charge', 'mass', 'm/z', 'retention_time', 'retention_length', 'ion_mobility_index', 'ion_mobility_length', 'num_isotopic_peaks']

        prototypes
            Prototypes with the columns: ['PrecursorID', 'Charge', 'MS1_Embedding', 'Proteins', 'Sequence', 'Run_Count', 'ESM_Embedding']

        use_anchors
            Whether to compute a relative representation.

        n_anchors
            Number of anchors that are the basis for the relative representations.

        Returns
        -------
            Representation for query and prototypes.
        """

        query_ms1_embeddings = query["ms1_embedding"]
        prototype_ms1_embeddings = prototypes["ms1_embedding"]

        if use_anchors:
            ms1_embeddings_means = np.mean(query_ms1_embeddings, axis=0)
            query_ms1_embeddings -= ms1_embeddings_means
            prototype_ms1_embeddings -= ms1_embeddings_means

            use_concatenated_embeddings = "ms2_embedding" in prototypes
            if use_concatenated_embeddings:
                prototype_embeddings = np.hstack(
                    (prototype_ms1_embeddings, prototypes["ms2_embedding"])
                )
            else:
                prototype_embeddings = prototype_ms1_embeddings

            query_anchors, prototype_anchors = self._select_anchors(
                query_anchor_candidates=prototype_ms1_embeddings,
                prototype_anchor_candidates=prototype_embeddings,
                n_anchors=n_anchors,
            )

            prototype_representation = self._compute_relative_representations(
                prototype_embeddings, prototype_anchors
            )

            query_representation = self._compute_relative_representations(
                query_ms1_embeddings, query_anchors
            )

            # Recover original embeddings. Keep operation in-place to save memory.
            query_ms1_embeddings += ms1_embeddings_means
            prototype_ms1_embeddings += ms1_embeddings_means

        else:
            prototype_representation = prototype_ms1_embeddings
            query_representation = query_ms1_embeddings

        return query_representation, prototype_representation

    def _select_anchors(
        self,
        query_anchor_candidates: np.ndarray,
        prototype_anchor_candidates: np.ndarray,
        n_anchors: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert query_anchor_candidates.shape[0] == prototype_anchor_candidates.shape[0]

        n_prototypes = query_anchor_candidates.shape[0]
        anchors_idx = np.random.choice(
            n_prototypes,
            size=n_anchors,
            replace=False,
        )

        query_anchors = query_anchor_candidates[anchors_idx]
        prototype_anchors = prototype_anchor_candidates[anchors_idx]

        return query_anchors, prototype_anchors

    def _find_k_closest_prototypes(
        self,
        query_representations: np.ndarray,
        prototype_representations: np.ndarray,
        k=5,
        distance_metric="euclidean",
    ):
        knn_index = NNDescent(prototype_representations, metric=distance_metric)
        neighbour_indices, distances = knn_index.query(query_representations, k=k)
        return neighbour_indices, distances

    def _compute_relative_representations(
        self, xs: np.ndarray, anchors: np.ndarray
    ) -> np.ndarray:
        """
        Compute relative representations.

        Parameters
        ----------
        xs
            Embeddings to transform into relative representations.

        anchors
            Anchor points used as the basis for the relative representation.

        Returns
        -------
            Relative representations of `xs`.
        """

        assert (
            xs.shape[1] == anchors.shape[1]
        ), "xs and anchors must have the same number of features"

        anchors_norm = np.linalg.norm(anchors, axis=1)
        xs_norm = np.linalg.norm(xs, axis=1)
        cosine_simularity = np.dot(anchors, xs.T) / np.outer(anchors_norm, xs_norm)

        relative_representation = cosine_simularity.T

        return relative_representation

    def _compute_affinities(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute affinities from distances.

        Parameters
        ----------
        distances
            Distances between query and prototypes.

        Returns
        -------
            Affinities between query and prototypes distances.
        """

        stds = np.mean(distances, axis=1)
        stds = stds**2
        stds = stds.reshape(-1, 1)
        affinities_tilda = np.exp(-np.true_divide(distances**2, stds))
        return affinities_tilda

    def _assign_prototype_to_query(
        self,
        prototype_probabilities: np.ndarray,
        prototype_indices: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assign prototype to query based on each candidate prototypes probability.

        Parameters
        ----------
        prototype_probabilities
            Probability query being assigned to each candidate prototype.

        prototype_indices
            Candidate prototypes indices in the complete prototype set.

        Returns
        -------
            Assigned prototype index and confidence score.
        """

        prototypes_subindex = np.argmax(prototype_probabilities, axis=1)
        confidence_scores = np.max(prototype_probabilities, axis=1)
        prototype_index = prototype_indices[
            np.arange(prototypes_subindex.shape[0]), prototypes_subindex
        ]

        return prototype_index, confidence_scores

    def _compute_assignment_probabilities(
        self,
        affinities: np.ndarray,
        charges: np.ndarray,
        target_charge: np.ndarray,
    ) -> np.ndarray:
        charge_mask = charges == target_charge.reshape(-1, 1)
        filtered_affinities = affinities * charge_mask

        denom = np.sum(filtered_affinities, axis=1).reshape(-1, 1)
        denom[denom == 0] = 1

        norm_affinities = filtered_affinities / denom

        return norm_affinities

    def _verify_ms_data(self, ms: pd.DataFrame):
        required_columns = [
            "Charge",
            "Mass",
            "m/z",
            "Retention time",
            "Retention length",
            "Ion mobility index",
            "Ion mobility length",
            "Number of isotopic peaks",
        ]

        for col in required_columns:
            assert col in ms.columns, f"Expected column {col} in the MS set."

    def _verify_msms_data(self, msms: pd.DataFrame):
        required_columns = [
            "PrecursorID",
            "Charge",
            "Mass",
            "m/z",
            "Retention time",
            "Retention length",
            "Ion mobility index",
            "Ion mobility length",
            "Number of isotopic peaks",
        ]

        for col in required_columns:
            assert col in msms.columns, f"Expected column {col} in the MSMS set."


class ESM2:
    """ESM-2 model for generating sequence embeddings.

    Available models:
    - "esm2_t6_8M_UR50D": 6 layers, 8 million parameters, 320-dim embedding
    - "esm2_t12_35M_UR50D": 12 layers, 35 million parameters, 480-dim embedding
    - "esm2_t30_150M_UR50D": 30 layers, 150 million parameters, 640-dim embedding
    - "esm2_t33_650M_UR50D": 33 layers, 650 million parameters, 1280-dim embedding
    - "esm2_t36_3B_UR50D": : 36 layers, 3 billion parameters, 2560-dim embedding, 6GB size
    - "esm2_t48_15B_UR50D": 48 layers, 15 billion parameters, 5120-dim embedding, 30GB size

    Note
    ----
        In the model names, the number after the "t" stands for the total layer number.
        The next number stands for the parameter number.
        "UR50D" stands for the used training dataset of the ESM-2 model which is UniRef 50.
    """

    def __init__(self, model_name: str = "esm2_t33_650M_UR50D"):
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.batch_converter = self.alphabet.get_batch_converter()

    def get_embeddings(
        self,
        sequences: List[str] | np.ndarray,
        representation_layer: int = 6,
        batch_size=256,
    ) -> np.ndarray:
        """
        Retrieve the embeddings for a list of sequences.

        Parameters
        ----------

        sequences
            List of sequences for which to retrieve the embeddings.

        representation_layer
            The n-th ESM-2 layer from which to retrieve the representations.

        Returns
        -------
            List of sequence embeddings.

        """
        tokenizer_input = list(enumerate(sequences))
        sequence_representations = []

        self.model.eval()
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch = tokenizer_input[i : i + batch_size]
            _, _, batch_tokens = self.batch_converter(batch)

            batch_length_mask = batch_tokens != self.alphabet.padding_idx
            batch_lens = batch_length_mask.sum(1)

            with torch.no_grad():
                results = self.model(
                    batch_tokens,
                    repr_layers=[representation_layer],
                    return_contacts=True,
                )

            token_representations = results["representations"][
                representation_layer
            ].cpu()

            # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
            for i, tokens_len in enumerate(batch_lens):
                sequence_representations.append(
                    token_representations[i, 1 : tokens_len - 1].mean(0)
                )

        return np.array(sequence_representations)


def _get_device(device: str | None = None) -> torch.device:
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return torch.device(device)
