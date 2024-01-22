import logging

from typing import Tuple, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from pynndescent import NNDescent
import esm
import umap
import matplotlib.pyplot as plt
import random

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
        prototypes_for_plm = []  # list of (proteins, sequence, species) for the PLM

        for group in precursor_groups:
            (precursor_id,) = group[0]
            locs = group[1].index
            charge = msms.loc[locs[0], "Charge"]
            ilocs = msms.index.get_indexer(locs)
            prototypes.append(
                (precursor_id, charge, np.mean(support_embeddings[ilocs], axis=0))
            )
            protein_name = group[1]['Proteins'].iloc[0]
            sequence = group[1]['Sequence'].iloc[0]
            species = group[1]['Species'].iloc[0]
            prototypes_for_plm.append((protein_name, sequence, species))

        '''
        # FIND SEQUENCES THAT ARE SHARED BETWEEN AT LEAST TWO SPECIES
        # create list of precursorIDs
        unique_precursor_ids = msms['PrecursorID'].drop_duplicates().tolist()

        # cut off charge at the end
        unique_precursor_ids_without_charge = []

        for original_string in unique_precursor_ids:
            # Find the index of the second '_'
            first_underscore_index = original_string.find('_')
            second_underscore_index = original_string.find('_', first_underscore_index + 1)

            # Cut off the end after the second '_'
            if second_underscore_index != -1:
                cut_off_string = original_string[:second_underscore_index]
                unique_precursor_ids_without_charge.append(cut_off_string)
            else:
                unique_precursor_ids_without_charge.append(original_string)

        # drop duplicates
        unique_precursor_ids_without_charge = list(set(unique_precursor_ids_without_charge))

        # find sequences that are shared among species
        result_strings = []

        for i, first_part in enumerate(unique_precursor_ids_without_charge):
            parts_i = first_part.split('_', 1)
            for j, second_part in enumerate(unique_precursor_ids_without_charge[i + 1:], start=i + 1):
                parts_j = second_part.split('_', 1)

                # Check if the substrings after the first '_' are the same
                if len(parts_i) > 1 and len(parts_j) > 1 and parts_i[1] == parts_j[1]:
                    result_strings.append(first_part)
                    result_strings.append(second_part)
                    print(first_part, second_part)

        print(result_strings)
        '''

        self._esm_call_list(prototypes_for_plm)

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

    '''
    Takes a list of tuples in the form ('Proteins', 'Sequence', 'Species').
    Forwards the sequences to the Potein Language Model ESM-2 and retrieves their representations.
    Reduces the representations with UMAP and plots them labeled by their species.
    Goal: Get representations that effectively separate sequences by their species.
    Todo:
    - Make sure that the order of sequences and their species are preserved.
    - Compare ESM-2 dimensions (small, medium, big) and layer numbers in particular.
    - Implement a metric to measure species separation. 
    - (Later:) Compare the ESM-2 output with the ProtT5 model.
    '''
    def _esm_call_list(self,
                       data_with_species: list[tuple[str, str, str]],
                       model_location="esm2_t6_8M_UR50D"
                       ):

        # Add true positive and false positive examples that are shared among all three species.
        true_positives = [
            ('tp1', '', ''),
            ('tp2', '', ''),
            ('tp3', '', ''),
            ('tp4', '', ''),
            ('tp5', '', '')
        ]

        false_positives = [
            ('fp1', '', ''),
            ('fp2', '', ''),
            ('fp3', '', ''),
            ('fp4', '', ''),
            ('fp5', '', '')
        ]

        # Add sequences that are shared between HeLa and Yeast.
        shared_sequences = [
            ('shared1', 'GLLLYGPPGTGK'),
            ('shared2', 'LQLWDTAGQER'),
            ('shared3', 'VAVVAGYGDVGK'),
            ('shared4', 'NMYQCQMGK'),
            ('shared5', 'YHPGYFGK'),
            ('shared6', 'DAHQSLLATR'),
            ('shared7', 'VPAINVNDSVTK'),
            ('shared8', 'VSTEVDAR'),
            ('shared9', 'MLSCAGADR'),
            ('shared10', 'LSDLLDWK'),
            ('shared11', 'QAVDVSPLR'),
            ('shared12', 'ETAEAYLGK'),
            ('shared13', 'STIGVEFATR'),
            ('shared14', 'YDCSSADINPIGGISK'),
            ('shared15', 'TTIFSPEGR'),
            ('shared16', 'QAVDVSPLRR'),
            ('shared17', 'ATAGDTHLGGEDFDNR'),
            ('shared18', 'RQAVDVSPLR'),
            ('shared19', 'VTILGHVQR'),
            ('shared20', 'FDNLYGCR'),
            ('shared21', 'GILFVGSGVSGGEEGAR'),
            ('shared22', 'FVIGGPQGDAGLTGR'),
            ('shared23', 'GVLMYGPPGTGK'),
            ('shared24', 'AQIWDTAGQER'),
            ('shared25', 'YENNVMNIR'),
            ('shared26', 'FPFAANSR')
        ]

        # Load ESM-2 model. It is currently predefined to download "esm2_t33_650M_UR50D".
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_location)
        batch_converter = alphabet.get_batch_converter()
        model.eval()  # disables dropout for deterministic results

        # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
        # use protein names or leading razor protein as identifier/tag
        # use unmodified sequence

        # Take only a subset of the data (duplicates allowed) due to extensive computation.
        subset_size = 1000
        random_subset_with_species = random.choices(data_with_species, k=subset_size)

        # Create copy with only the 'Name' and 'Sequence' columns and leave the 'Species' column out.
        random_subset_without_species = [(x[0], x[1]) for x in random_subset_with_species]

        # Initialize list for ESM-2 representations
        sequence_representations = []

        # Iterate through the list in bulks because the execution is more stable that way.
        bulk_size = 500
        for i in range(0, len(random_subset_without_species), bulk_size):
            print("i:", i)
            bulk = random_subset_without_species[i:i + bulk_size]

            # Use batch converter as done in ESM-2 example.
            batch_labels, batch_strs, batch_tokens = batch_converter(bulk)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

            # Extract per-residue representations (on CPU) as done in ESM-2 example.
            # Todo: try different layer numbers and fit them to the input model
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[6], return_contacts=True)
            token_representations = results["representations"][6]

            # Generate per-sequence representations via averaging as done in ESM-2 example.
            # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
            for i, tokens_len in enumerate(batch_lens):
                sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))

        # Compute example representations of shared sequences.
        example_representations = []

        batch_labels, batch_strs, batch_tokens = batch_converter(shared_sequences)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[6], return_contacts=True)
        token_representations = results["representations"][6]

        for i, tokens_len in enumerate(batch_lens):
            example_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))

        # Reduce ESM-2 representation to two-dimensional UMAP-embedding.
        reducer = umap.UMAP(metric='cosine')
        umap_embedding = reducer.fit_transform(sequence_representations)
        example_umap_embedding = reducer.fit_transform(example_representations)

        # Plot UMAP-embedding without species.
        fig, ax = plt.subplots(figsize=(14, 12), ncols=1, nrows=1)
        sp = ax.scatter(umap_embedding[:, 0], umap_embedding[:, 1], s=0.1)
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_title('PLM embedding')
        fig.colorbar(sp)
        plt.show()

        # Plot UMAP-embedding of example sequences that are shared between HeLa and Yeast.
        fig, ax = plt.subplots(figsize=(14, 12), ncols=1, nrows=1)
        ax.scatter(umap_embedding[:, 0], umap_embedding[:, 1], s=0.01, color='red')
        sp = ax.scatter(example_umap_embedding[:, 0], example_umap_embedding[:, 1], s=0.1)
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_title('PLM embedding')
        fig.colorbar(sp)
        plt.show()

        # Connect UMAP-embedding with species label.
        # Todo: make sure that the order is preserved.
        umap_with_species = []
        for i in range(len(umap_embedding)):
            umap_with_species.append((umap_embedding[i], random_subset_with_species[i][2]))

        # Prepare plot of UMAP-embedding with species.
        colors = ['red', 'blue', 'green']
        color_map = {species: color for species, color in zip(('HeLa', 'Yeast', 'Ecoli'), colors)}
        species_size = {'HeLa': .1, 'Yeast': .4, 'Ecoli': 1.4}
        fig, ax = plt.subplots(figsize=(14, 14))
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_title('PLM prototypes | latent space - Species')

        # Plot UMAP-embedding with species.
        for species, color in color_map.items():
            x_data = [x[0] for x in umap_with_species if x[1] == species]
            if len(x_data) > 0:
                umap1, umap2 = zip(*x_data)
                ax.scatter(umap1, umap2, c=color, s=species_size[species], label=species, alpha=0.6)

        ax.legend()
        plt.show()
