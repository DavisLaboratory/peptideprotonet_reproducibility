import logging

from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pynndescent import NNDescent
import esm
import torch
import scanpy

from typing import Any

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
        esm2_model="esm2_t33_650M_UR50D",
        use_precomputed_esm_embeddings=False,
        esm_embedding_path="example_data/embeddings_esm2_t6_8M_UR50D_layer6_datasetHeLa_Yeast_Ecoli.npy",
        n_anchors=300,
        n_principal_components=10,
        n_representation_layer=6,
        use_preselected_anchors=False,
        anchors_idx=[],
        use_concatenated_embeddings=True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate the identities/labels from the support set to the query set.

        Parameters
        ----------
        ms
            Query set with the columns: ['Charge', 'Mass', 'm/z', 'Retention time', 'Retention length', 'Ion mobility index', 'Ion mobility length', 'Number of isotopic peaks']

        msms
            Support set from which the prototypes are taken.
            with the columns: ['PrecursorID', 'Charge', 'Mass', 'm/z', 'Retention time', 'Retention length', 'Ion mobility index', 'Ion mobility length', 'Number of isotopic peaks']

        k_neighbours
            Number of neighbours to consider when computing identities and confidence.

        verbose
            Whether to print progress.

        esm2_model
            Name of the ESM2 model to use.

        use_precomputed_esm_embeddings
            Whether to use precomputed ESM-2 embeddings. Path needs to be specified in 'esm_embedding_path'.

        esm_embedding_path
            Path to the precomputed ESM-2 embeddings file.

        n_anchors
            Number of anchors that are the basis for the relative representations.

        n_principal_components
            Number of principal components to which the ESM-2 embeddings are reduced.

        n_representation_layer
            The n-th ESM-2 layer of which to retrieve the embeddings.

        generate_visualizations
            Whether to generate visualizations.

        use_preselected_anchors
            Whether to use preselected anchors for the relative representations. If true, then they need to be
            specified in anchors_idx

        anchors_idx
            Fixed anchor indexes for the relative representations.

        use_concatenated_embeddings
            Whether to use the concatenated MS1+ESM embedding for the prototypes. If false, then just the
            MS1 embedding will be used for the prototypes.

        dataset
            Whether this function is calles from the "HeLa_Yeast_Ecoli" or "HeLa" notebook/dataset.

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

        # returns prototypes: ['PrecursorID', 'Charge', 'MS1_Embedding', 'Proteins', 'Sequence', 'Species', 'Run_Count', 'ESM_Embedding']
        # makes call to ESM-2 to compute the sequence embeddings of prototypes.
        prototypes = self._compute_prototypes(
            msms,
            verbose=verbose,
            esm2_model=esm2_model,
            #use_precomputed_esm_embeddings=use_precomputed_esm_embeddings,
            #esm_embedding_path=esm_embedding_path,
            esm2_representation_layer=n_representation_layer,
        )

        identities, confidence = self._propagate_using_prototypes(
            ms,
            prototypes,
            k_neighbours=k_neighbours,
            use_anchors=use_anchors,
            verbose=verbose,
            n_anchors=n_anchors,
            n_principal_components=n_principal_components,
            use_preselected_anchors=use_preselected_anchors,
            anchors_idx=anchors_idx,
            use_concatenated_embeddings=use_concatenated_embeddings,
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
        n_anchors=300,
        n_principal_components=10,
        use_preselected_anchors=False,
        anchors_idx=[],
        use_concatenated_embeddings=True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Helper function to propagate identities using prototypes.

        Parameters
        ----------
        MS
            Query set with the columns: ['charge', 'mass', 'm/z', 'retention_time', 'retention_length', 'ion_mobility_index', 'ion_mobility_length', 'num_isotopic_peaks']

        prototypes
            Prototypes with the columns: ['PrecursorID', 'Charge', 'MS1_Embedding', 'Proteins', 'Sequence', 'Species', 'Run_Count', 'ESM_Embedding']

        k_neighbours
            Number of neighbours to consider when computing identities and confidence.

        distance_metric
            Distance metric to use when computing neighbours.

        use_anchors
            Whether to compute a relative representation.

        verbose
            Whether to print progress.

        n_anchors
            Number of anchors that are the basis for the relative representations.

        n_principal_components
            Number of principal components to which the ESM-2 embeddings are reduced.

        Returns
        -------
            Predicted identities and confidence.
        """

        # Get MS1 embedding of query precursors
        query_embeddings = self.get_latent_representations(ms[self._features])
        query_charges = ms["Charge"].values

        print("Computing relative representations...")

        if use_anchors:
            means = np.mean(query_embeddings, axis=0)
            query_embeddings -= means
            prototypes["MS1_Embedding"] -= means

            if use_concatenated_embeddings:
                # Retrieve the first n principal components of the ESM-2 embeddings.
                principal_components_esm = scanpy.pp.pca(
                    prototypes["ESM_Embedding"], n_comps=n_principal_components
                )
                prototypes["Concatenated_Embedding"] = np.hstack(
                    (prototypes["MS1_Embedding"], principal_components_esm)
                )

                """
                1. Filters out prototypes that are not from the species 'HeLa'. 
                   Should be generalized on the species from the query data set in the future.
                2. Either:
                    a. Selects a random subset of the filtered prototypes as anchors.
                    b. Selects the prototypes of which the indexes were already specified in 'anchors_idx'.
                3. Returns anchor prototypes.
                """
                anchors = self._select_anchors(
                    prototypes,
                    n_anchors=n_anchors,
                    use_preselected_anchors=use_preselected_anchors,
                    preselected_anchors_idx=anchors_idx,
                    use_concatenated_embeddings=use_concatenated_embeddings,
                )

                # Compute Relative Representations of Prototypes based on Concatenated MS1-ESM Embeddings
                prototype_representation = self._compute_relative_representations(
                    prototypes["Concatenated_Embedding"],
                    anchors["Concatenated_Embedding"],
                )

                # Compute Relative Representations of Query Precursors based on MS1 Embeddings
                query_representation = self._compute_relative_representations(
                    query_embeddings, anchors["MS1_Embedding"]
                )

            else:
                anchors = self._select_anchors(
                    prototypes,
                    n_anchors=n_anchors,
                    use_preselected_anchors=use_preselected_anchors,
                    preselected_anchors_idx=anchors_idx,
                    use_concatenated_embeddings=use_concatenated_embeddings,
                )

                # Compute Relative Representations of Prototypes based on MS1 Embeddings
                prototype_representation = self._compute_relative_representations(
                    prototypes["MS1_Embedding"], anchors["MS1_Embedding"]
                )

                # Compute Relative Representations of Query Precursors based on MS1 Embeddings
                query_representation = self._compute_relative_representations(
                    query_embeddings, anchors["MS1_Embedding"]
                )

            # recover original embeddings - keep in-place to save memory
            query_embeddings += means
            prototypes["MS1_Embedding"] += means

        else:
            # Without Relative Representations: MS1 Embedding is the representation. ESM-2 embeddings are not used here.
            prototype_representation = prototypes["MS1_Embedding"]
            query_representation = query_embeddings

        print("setting up nearest neighbour graph...")

        knn_index = NNDescent(prototype_representation, metric=distance_metric)

        if verbose:
            print(
                f"computing {k_neighbours}-nearest neighbour prototypes of query precursors..."
            )

        neighbours, distances = knn_index.query(query_representation, k=k_neighbours)

        neighbours_weights = self._compute_weights(distances)
        neighbours_charges = np.array(
            [prototypes["Charge"][q_neighbours] for q_neighbours in neighbours]
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
        self,
        prototypes: np.ndarray,
        n_anchors=5,
        use_preselected_anchors=False,
        preselected_anchors_idx=[],
        use_concatenated_embeddings=True,
    ) -> np.ndarray:
        """
        Select anchors from the latent embeddings using uniform sampling.

        Parameters
        ----------
        latent_embeddings
            Latent embeddings to consider when selecting anchors.

        concatenated_embeddings
            Concatenated embeddings to consider when selecting anchors.

        n_anchors
            Number of anchors to select.

        Returns
        -------
            Anchors.
        """

        """
        Possible Todo: Select anchors only from peptides that were present in 9/10 or 10/10 runs.
        The way it was done with the option preselect_most_present_anchors did not improve results.
        """
        preselect_most_present_anchors = False
        if preselect_most_present_anchors:
            # filter out peptides that were in too few runs
            filtered_latent_embeddings = prototypes["MS1_Embedding"][
                (
                    (prototypes["Run_Count"] > 5)
                    & (
                        (prototypes["Species"] == "Ecoli")
                        | (prototypes["Species"] == "Yeast")
                    )
                )
                | ((prototypes["Run_Count"] > 15) & (prototypes["Species"] == "HeLa"))
            ]
            filtered_concatenated_embeddings = prototypes["Concatenated_Embedding"][
                (
                    (prototypes["Run_Count"] > 5)
                    & (
                        (prototypes["Species"] == "Ecoli")
                        | (prototypes["Species"] == "Yeast")
                    )
                )
                | ((prototypes["Run_Count"] > 15) & (prototypes["Species"] == "HeLa"))
            ]

            anchors_idx = np.random.choice(
                filtered_latent_embeddings.shape[0], size=n_anchors, replace=False
            )
            anchors = {
                "MS1_embedding": filtered_latent_embeddings[anchors_idx],
                "Concatenated_embedding": filtered_concatenated_embeddings[anchors_idx],
            }

        else:
            # prefilter prototypes for species from query data set (here: HeLa)
            right_species_indices = np.where(prototypes["Species"] == "HeLa")[0]
            filtered_prototypes = {
                key: value[right_species_indices] for key, value in prototypes.items()
            }

            if use_preselected_anchors:
                # Take anchors from preselected_anchors_idx.
                if use_concatenated_embeddings:
                    # Store coupled (MS1, MS1+ESM) anchors.
                    anchors = {
                        "MS1_Embedding": filtered_prototypes["MS1_Embedding"][
                            preselected_anchors_idx
                        ],
                        "Concatenated_Embedding": filtered_prototypes[
                            "Concatenated_Embedding"
                        ][preselected_anchors_idx],
                    }
                else:
                    # Store simple (MS1) anchors. Here, the ESM-embedding is not used.
                    anchors = {
                        "MS1_Embedding": filtered_prototypes["MS1_Embedding"][
                            preselected_anchors_idx
                        ]
                    }
            else:
                # Select a set of random anchor points.
                random_state = np.random.RandomState()
                anchors_idx = random_state.choice(
                    filtered_prototypes["MS1_Embedding"].shape[0],
                    size=n_anchors,
                    replace=False,
                )

                if use_concatenated_embeddings:
                    # Store coupled (MS1, MS1+ESM) anchors
                    anchors = {
                        "MS1_Embedding": filtered_prototypes["MS1_Embedding"][
                            anchors_idx
                        ],
                        "Concatenated_Embedding": filtered_prototypes[
                            "Concatenated_Embedding"
                        ][anchors_idx],
                    }
                else:
                    # Store simple (MS1) anchors. Here, the ESM-embedding is not used.
                    anchors = {
                        "MS1_Embedding": filtered_prototypes["MS1_Embedding"][
                            anchors_idx
                        ]
                    }

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

        anchors_norm = np.linalg.norm(anchors, axis=1)
        xs_norm = np.linalg.norm(xs, axis=1)
        cosine_simularity = np.dot(anchors, xs.T) / np.outer(anchors_norm, xs_norm)
        relative_representation = cosine_simularity.T

        return relative_representation

    def _compute_prototypes(
        self,
        msms: pd.DataFrame,
        esm2_model = "esm2_t33_650M_UR50D",
        esm2_representation_layer=6,
        verbose=False,
    ) -> Dict[str, np.ndarray]:
        """
        Helper function to compute prototypes.

        Parameters
        ----------
        msms
            Support set with the columns: ['PrecursorID', 'Charge', 'Mass', 'm/z', 'Retention time', 'Retention length', 'Ion mobility index', 'Ion mobility length', 'Number of isotopic peaks']

        esm2_model
            Name of the ESM-2 model.

        esm2_representation_layer
            The n-th ESM-2 layer of which to retrieve the embeddings.

        verbose
            Whether to print progress.

        Returns
        -------
            Prototypes with the columns: ['PrecursorID', 'Charge', 'MS1_Embedding', 'Proteins', 'Sequence', 'Species', 'Run_Count', 'ESM_Embedding']
        """

        if verbose:
            print("computing prototypes...")

        # Gets MS1 embedding of prototypes
        support_embeddings = self.get_latent_representations(msms[self._features])

        # list of (precursor_id, charge, ms1_embedding, proteins, sequence, species, run_count, esm_embedding)
        prototypes = []
        precursor_groups = msms.groupby(["PrecursorID"])

        for group in precursor_groups:
            (precursor_id,) = group[0]
            locs = group[1].index
            charge = msms.loc[locs[0], "Charge"]
            ilocs = msms.index.get_indexer(locs)
            proteins = group[1]["Proteins"].iloc[0]
            sequence = group[1]["Sequence"].iloc[0]
            if "Species" in group[1].columns:
                species = group[1]["Species"].iloc[0]
            else:  # only applicable for the HeLa example notebook
                species = "HeLa"
            run_count = len(group[1])
            prototypes.append(
                (
                    precursor_id,
                    charge,
                    np.mean(support_embeddings[ilocs], axis=0),
                    proteins,
                    sequence,
                    species,
                    run_count,
                )
            )

        (
            precursor_ids,
            charges,
            ms1_embeddings,
            proteins,
            sequences,
            species,
            run_count,
        ) = zip(*prototypes)

        precursor_ids = np.array(precursor_ids)
        charges = np.array(charges)
        ms1_embeddings = np.array(ms1_embeddings)
        proteins = np.array(proteins)
        sequences = np.array(sequences)
        species = np.array(species)
        run_count = np.array(run_count)

        prototypes = {
            "PrecursorID": precursor_ids,
            "Charge": charges,
            "MS1_Embedding": ms1_embeddings,
            "Protein": proteins,
            "Sequence": sequences,
            "Species": species,
            "Run_Count": run_count,
        }

        esm_embeddings = self._compute_prototype_embeddings(prototypes, esm2_model, esm2_representation_layer)
           
        prototypes["ESM_Embedding"] = esm_embeddings

        return prototypes

    def _compute_prototype_embeddings(
        self,
        prototypes: dict[str, Any],
        esm2_model: str = "esm2_t33_650M_UR50D",
        representation_layer: int = 6,
    ) -> np.ndarray:
        model, alphabet = esm.pretrained.load_model_and_alphabet(esm2_model)
        batch_converter = alphabet.get_batch_converter()

        model.eval()

        # TODO: robustness
        # prepare input for ESM as list of tuples (PrecursorID, Sequence).
        esm_input = list(zip(prototypes["PrecursorID"], prototypes["Sequence"]))

        esm_embeddings = self._get_esm_embedding(
            batch_converter, model, alphabet, esm_input, representation_layer
        )

        return np.array(esm_embeddings)
        np.save(
            f"example_data/embeddings_{esm2_model}_layer{representation_layer}_dataset{dataset}",
            esm_embeddings_np,
        )

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

        charge_mask = neighbours_charges == query_charges.reshape(-1, 1)
        filtered_weights = neighbours_weights * charge_mask

        denominator = np.sum(filtered_weights, axis=1).reshape(-1, 1)
        denominator[denominator == 0] = 1

        # normalize confidences based on charges between neighbour and query matching
        probs = filtered_weights / denominator

        predictions = np.argmax(probs, axis=1)
        confidence = np.max(probs, axis=1)

        return predictions, confidence

    def _get_esm_embedding(
        self,
        batch_converter,
        model,
        alphabet,
        data,
        representation_layer: int,
        batch_size = 512,
    ) -> np.ndarray:
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

        for i in range(0, len(data), batch_size):
            print("ESM-2 embeddings of peptide sequences already computed:", i)
            batch = data[i : i + batch_size]

            # Use batch converter as recommended by ESM-2 documentation.
            batch_labels, batch_strs, batch_tokens = batch_converter(batch)

            batch_length_mask = (batch_tokens != alphabet.padding_idx)
            batch_lens = batch_length_mask.sum(1)

            # Extract per-residue representations (on CPU) as recommended by ESM-2 documentation.
            # n_representation_layer specifies the ESM-2 layer from which to take the representation.
            with torch.no_grad():
                results = model(
                    batch_tokens,
                    repr_layers=[representation_layer],
                    return_contacts=True,
                )

            token_representations = results["representations"][representation_layer].cpu()

            # Generate per-sequence representations via averaging as recommended by ESM-2 documentation.
            # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
            for i, tokens_len in enumerate(batch_lens):
                sequence_representations.append(
                    token_representations[i, 1 : tokens_len - 1].mean(0)
                )

        return np.array(sequence_representations)
