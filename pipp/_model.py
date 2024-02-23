import logging

from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pynndescent import NNDescent
import esm
import umap
import matplotlib.pyplot as plt
import random
import csv
import pickle
from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
from scipy.spatial.distance import euclidean

from ._module import Encoder

logger = logging.getLogger(__name__)

'''
Files to generate:
- example_data/esm2_t33_650M_UR50D_embeddings.npy
- example_data/all_prototypes.npz
- example_data/protT5_embeddings_df.pkl
- example_data/final_false_list.pkl
- example_data/final_true_list.pkl
'''


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

        '''
        Between prototype computation and propagation:
        1. Get ESM-2 embedding (1280-dimensional) of all prototypes
        2. Concatenate MS1 embedding and ESM-2 embedding of all prototypes
        
        In the propagation process:
        1. Define (e.g. 20 or 100) random anchor points from the prototypes
        2. Compute similarity of all prototypes to the anchor points as a new (20- or 100-dimensional) embedding
           by comparing their concatenated MS1+ESM-2 embedding
        3. Compute similarity of all MS1 data points to the anchor points as a new (20- or 100-dimensional) embedding
           by comparing their MS1 embedding
        4. Use the similarity embeddings as the basis for the PyNNDescent computation
        '''

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
            Prototypes with the columns: ['precursor_id', 'charge', 'embedding', 'protein', 'sequence', 'species']

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

        '''
        # Show example false transfer prototype sequences and true transfer prototype sequences and their nearest neighbours
        false_positives = [
            ('Ecoli_LPYITFPEGSEEHTYLHAQR_4', 'LPYITFPEGSEEHTYLHAQR')
            ,('Yeast_LSLTGGFSHHHATDDVEDAAPETK_4', 'LSLTGGFSHHHATDDVEDAAPETK')
            ,('Ecoli_ILEVLQEPDNHHVSAEDLYK_4', 'ILEVLQEPDNHHVSAEDLYK')
            ,('Ecoli_DVSLLHKPTTQISDFHVATR_4', 'DVSLLHKPTTQISDFHVATR')
            ,('Yeast_FGLPHADDVLGLPIGQHIVIK_4', 'FGLPHADDVLGLPIGQHIVIK')
            ,('Yeast_TLLEAIDAIEQPSRPTDKPLR_4', 'TLLEAIDAIEQPSRPTDKPLR')
            ,('Yeast_LMGLYLPDGGHLSHGYATENR_4', 'LMGLYLPDGGHLSHGYATENR')
            ,('Yeast_GDAGHPSIATTHNHSTSK_3', 'GDAGHPSIATTHNHSTSK')
            ,('Yeast_STSGNTHLGGQDFDTNLLEHFK_4', 'STSGNTHLGGQDFDTNLLEHFK')
            ,('Yeast_VKPTVNQVETHPHLPQMELR_4', 'VKPTVNQVETHPHLPQMELR')
            ,('Yeast_ITLDTDKLPPHTQIFQAGTK_3', 'ITLDTDKLPPHTQIFQAGTK')
            ,('Ecoli_ALYPCPLHGISEDDAIASIHR_4', 'ALYPCPLHGISEDDAIASIHR')
            ,('Yeast_EGDDAPESPDIHFEPVVHLEK_4', 'EGDDAPESPDIHFEPVVHLEK')
            ,('Yeast_SIQEHPDKNPNDPTATER_3', 'SIQEHPDKNPNDPTATER')
            ,('Yeast_AADTPETSDAVHTEQKPEEEK_3', 'AADTPETSDAVHTEQKPEEEK')
            ,('Yeast_ETNKPHTETITSVEPTNK_3', 'ETNKPHTETITSVEPTNK')
            ,('Yeast_YAGEVSHDDK_2', 'YAGEVSHDDK')
            ,('Yeast_SIVPSGASTGVHEALEMRDEDK_4', 'SIVPSGASTGVHEALEMRDEDK')
            ,('Yeast_FELSGIPPAPR_2', 'FELSGIPPAPR')
            ,('Yeast_TSSGNEMPPQDAEGWFYK_2', 'TSSGNEMPPQDAEGWFYK')
            ,('Yeast_AAVDCECEFQNLEHNEK_3', 'AAVDCECEFQNLEHNEK')
            ,('Yeast_EVETEKEEVKEDDSK_3', 'EVETEKEEVKEDDSK')
            ,('Yeast_TVEEDHPIPEDVHENYENK_4', 'TVEEDHPIPEDVHENYENK')
            ,('Yeast_GSDYDYNNSTHSAEHTPR_3', 'GSDYDYNNSTHSAEHTPR')
            ,('Yeast_HGQSEWNEK_2', 'HGQSEWNEK')
            ,('Yeast_LHQDQQGQDNAAVHLTLK_3', 'LHQDQQGQDNAAVHLTLK')
            ,('Yeast_EPQHQAAVPVSQEENER_3', 'EPQHQAAVPVSQEENER')
            ,('Yeast_GDIIIDGGNSHFPDSNRR_3', 'GDIIIDGGNSHFPDSNRR')
            ,('Yeast_EEQQNNQATAGEHDASITR_3', 'EEQQNNQATAGEHDASITR')
            ,('Yeast_QTVIIAHYPPTVQAGEATK_3', 'QTVIIAHYPPTVQAGEATK')
            ,('Ecoli_GATIVGHWPTAGYHFEASK_3', 'GATIVGHWPTAGYHFEASK')
            ,('Yeast_NVIAETGAGQHGVATATACAK_3', 'NVIAETGAGQHGVATATACAK')
            ,('Yeast_LISEAESHFSQGNHAEAVAK_3', 'LISEAESHFSQGNHAEAVAK')
            ,('Yeast_QELEEQAHQAQLDHEQQITQVK_3', 'QELEEQAHQAQLDHEQQITQVK')
            ,('Yeast_LHSHDHKPPVSESSDWQK_3', 'LHSHDHKPPVSESSDWQK')
            ,('Yeast_LSICTCDGEDHPNQGVGR_3', 'LSICTCDGEDHPNQGVGR')
            ,('Yeast_VQQQQLQQAQAQQQANR_3', 'VQQQQLQQAQAQQQANR')
            ,('Yeast_HFDGAHGVVVPR_3', 'HFDGAHGVVVPR')
            ,('Yeast_VQLLTPCLHMLPADHFGFK_4', 'VQLLTPCLHMLPADHFGFK')
            ,('Yeast_VPSDSSGPVGVCTYDDHR_3', 'VPSDSSGPVGVCTYDDHR')
            ,('Yeast_GHPYYVGTQYHPEYTSK_3', 'GHPYYVGTQYHPEYTSK')
        ]

        true_positives = [
            ('HeLa_LDNVLLDSEGHIK_3', 'LDNVLLDSEGHIK'),
            ('HeLa_PLRLPLQDVYK_3', 'PLRLPLQDVYK'),
            ('HeLa_SCPSEVLVCTTSPDRPGPPTRPLVK_4', 'SCPSEVLVCTTSPDRPGPPTRPLVK'),
            ('HeLa_LLGHWEEAAHDLALACK_4', 'LLGHWEEAAHDLALACK'),
            ('HeLa_SISFHPSGDFILVGTQHPTLR_4', 'SISFHPSGDFILVGTQHPTLR'),
            ('HeLa_FDVHDVTLHADAIHR_4', 'FDVHDVTLHADAIHR'),
            ('HeLa_LLMHLEEMQHTISTDEEK_4', 'LLMHLEEMQHTISTDEEK'),
            ('HeLa_PYNFLAHGVLPDSGHLHPLLK_4', 'PYNFLAHGVLPDSGHLHPLLK'),
            ('HeLa_FNGGGHINHSIFWTNLSPNGGGEPK_4', 'FNGGGHINHSIFWTNLSPNGGGEPK'),
            ('HeLa_AIGISNFNHLQVEMILNKPGLK_4', 'AIGISNFNHLQVEMILNKPGLK'),
            ('HeLa_RILEDQEENPLPAALVQPHTGK_4', 'RILEDQEENPLPAALVQPHTGK'),
            ('HeLa_KYEDICPSTHNMDVPNIKR_4', 'KYEDICPSTHNMDVPNIKR'),
            ('HeLa_NLYHNLCTSLFPTIHGNDEVK_4', 'NLYHNLCTSLFPTIHGNDEVK'),
            ('HeLa_ILAGDVETHAEMVHSAFQAQR_4', 'ILAGDVETHAEMVHSAFQAQR'),
            ('HeLa_LQSNPSLEGVSHVIVDEVHER_4', 'LQSNPSLEGVSHVIVDEVHER'),
            ('HeLa_ICANHYITPMMELKPNAGSDR_4', 'ICANHYITPMMELKPNAGSDR'),
            ('HeLa_GQHVTGSPFQFTVGPLGEGGAHK_4', 'GQHVTGSPFQFTVGPLGEGGAHK'),
            ('HeLa_EAEAAIYHLQLFEELRR_4', 'EAEAAIYHLQLFEELRR'),
            ('HeLa_GAEGILAPQPPPPQQHQERPGAAAIGSAR_4', 'GAEGILAPQPPPPQQHQERPGAAAIGSAR'),
            ('HeLa_VGLYDTYSNKPPQISSTYHK_4', 'VGLYDTYSNKPPQISSTYHK'),
            ('HeLa_FVHSENQHLVSPEALDFLDK_4', 'FVHSENQHLVSPEALDFLDK'),
            ('HeLa_DIELHLESSSHQETLDHIQK_4', 'DIELHLESSSHQETLDHIQK'),
            ('HeLa_YFLQATHVQPDDIGAHMNVGR_4', 'YFLQATHVQPDDIGAHMNVGR'),
            ('HeLa_LSSLIILMPHHVEPLER_4', 'LSSLIILMPHHVEPLER'),
            ('HeLa_QLQAAAAHWQQHQQHR_3', 'QLQAAAAHWQQHQQHR'),
            ('HeLa_LCHITSGEALPLDHTLETWIAK_4', 'LCHITSGEALPLDHTLETWIAK'),
            ('HeLa_SQSAAVTPSSTTSSTR_2', 'SQSAAVTPSSTTSSTR'),
            ('HeLa_KDQVTAQEIFQDNHEDGPTAK_4', 'KDQVTAQEIFQDNHEDGPTAK'),
            ('HeLa_PNWDYHAEIQAFGHR_4', 'PNWDYHAEIQAFGHR'),
            ('HeLa_EVKPEETTCSEHCLQK_3', 'EVKPEETTCSEHCLQK'),
            ('HeLa_SPVYSHFNETLLGVSVIR_3', 'SPVYSHFNETLLGVSVIR'),
            ('HeLa_PGGVVHSFSHNVGPGDK_4', 'PGGVVHSFSHNVGPGDK'),
            ('HeLa_KHEAFESDLAAHQDR_4', 'KHEAFESDLAAHQDR'),
            ('HeLa_QEAGISEGQGTAGEEEEKK_3', 'QEAGISEGQGTAGEEEEKK'),
            ('HeLa_HSVVAGGGGGEGR_2', 'HSVVAGGGGGEGR'),
            ('HeLa_ERVEAGDVIYIEANSGAVK_3', 'ERVEAGDVIYIEANSGAVK'),
            ('HeLa_DVHNIYGLYVHMATADGLR_4', 'DVHNIYGLYVHMATADGLR'),
            ('HeLa_HQPWQSPER_2', 'HQPWQSPER'),
            ('HeLa_HEMPPHIYAITDTAYR_3', 'HEMPPHIYAITDTAYR'),
            ('HeLa_LYCQTTGLGGSAVAGHASDK_3', 'LYCQTTGLGGSAVAGHASDK'),
            ('HeLa_LLQDSVDFSLADAINTEFK_3', 'LLQDSVDFSLADAINTEFK')
        ]

        # Retrieve indexes of example prototypes in the ndarray of all prototypes
        false_positive_indexes = []
        for false_positive in false_positives:
            index = np.where(prototypes['PrecursorID'] == false_positive[0])[0][0]
            false_positive_indexes.append(index)

        true_positive_indexes = []
        for true_positive in true_positives:
            index = np.where(prototypes['PrecursorID'] == true_positive[0])[0][0]
            true_positive_indexes.append(index)
        '''

        query_embeddings = self.get_latent_representations(ms[self._features])
        query_charges = ms["Charge"].values

        prototype_embeddings = prototypes["Embedding"]
        prototype_charges = prototypes["Charge"]

        print('Computing relative representations...')

        if use_anchors:
            means = np.mean(query_embeddings, axis=0)
            query_embeddings -= means
            prototype_embeddings -= means

            concatenated_embedding = np.hstack((prototype_embeddings, prototypes['Esm_embedding']))

            anchors = self._select_anchors(prototype_embeddings, concatenated_embedding)

            use_concatenated_embeddings = True
            if use_concatenated_embeddings:
                prototype_representation = self._compute_relative_representations(
                    concatenated_embedding, anchors['Concatenated_embedding']
                )
                query_representation = self._compute_relative_representations(
                    query_embeddings, anchors['MS1_embedding']
                )

            else:
                prototype_representation = self._compute_relative_representations(
                    prototype_embeddings, anchors['MS1_embedding']
                )
                query_representation = self._compute_relative_representations(
                    query_embeddings, anchors['MS1_embedding']
                )

            # plot the relative representations of the prototypes
            reducer = umap.UMAP(metric='cosine')
            umap_embedding = reducer.fit_transform(prototype_representation)
            fig, ax = plt.subplots(figsize=(14, 14))
            ax.set_xlabel('UMAP1')
            ax.set_ylabel('UMAP2')
            ax.set_title('Prototype relative representations')
            umap1 = umap_embedding[:, 0]
            umap2 = umap_embedding[:, 1]
            ax.scatter(umap1, umap2, c='blue', s=.1, label='Prototype', alpha=.5)
            ax.legend(markerscale=8)
            plt.show()

            # plot the relative representations of the MS1 datapoints
            umap_embedding = reducer.fit_transform(query_representation)
            fig, ax = plt.subplots(figsize=(14, 14))
            ax.set_xlabel('UMAP1')
            ax.set_ylabel('UMAP2')
            ax.set_title('MS1 relative representations')
            umap1 = umap_embedding[:, 0]
            umap2 = umap_embedding[:, 1]
            ax.scatter(umap1, umap2, c='green', s=.1, label='MS1 datapoint', alpha=.5)
            ax.legend(markerscale=8)
            plt.show()


            # recover original embeddings - keep in-place to save memory
            query_embeddings += means
            prototype_embeddings += means

        else:
            prototype_representation = prototype_embeddings
            query_representation = query_embeddings

        knn_index = NNDescent(
            prototype_representation, metric=distance_metric
        )

        '''
        # Get neighbour indexes of true/false transfer prototype examples
        false_neighbours, false_distances = knn_index.query(prototype_representation[false_positive_indexes], k=5)
        true_neighbours, true_distances = knn_index.query(prototype_representation[true_positive_indexes], k=5)

        # Get neighbour PrecursorIDs
        false_neighbour_ids = prototypes['PrecursorID'][false_neighbours]
        true_neighbour_ids = prototypes['PrecursorID'][true_neighbours]

        # Get ESM-2 embeddings of example prototypes and their neighbours.
        # self._esm_call_examples("esm2_t6_8M_UR50D", false_positives, true_positives, false_neighbour_ids, true_neighbour_ids)
        '''

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
        self, latent_embeddings: np.ndarray, concatenated_embeddings: np.ndarray, n_anchors=100
    ) -> np.ndarray:
        """
        Select anchors from the latent embeddings using uniform sampling.

        Parameters
        ----------
        latent_embeddings
            Latent embeddings to consider when selecting anchors.

        concatenated_embeddings

        n_anchors
            Number of anchors to select.

        Returns
        -------
            Anchors.
        """

        anchors_idx = np.random.choice(
            latent_embeddings.shape[0], size=n_anchors, replace=False
        )
        anchors = {'MS1_embedding': latent_embeddings[anchors_idx],
                    'Concatenated_embedding': concatenated_embeddings[anchors_idx]}

        # why do we use a copy?
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

        metric = "euclidean"

        if metric == "cosine":
            anchors_norm = np.linalg.norm(anchors, axis=1)
            xs_norm = np.linalg.norm(xs, axis=1)
            cosine_simularity = np.dot(anchors, xs.T) / np.outer(anchors_norm, xs_norm)
            relative_representation = cosine_simularity.T

        if metric == "euclidean":
            relative_representation = np.empty((xs.shape[0], anchors.shape[0]))

            for element in range(xs.shape[0]):

                similarity_array = np.empty((anchors.shape[0]))

                for anchor in range(anchors.shape[0]):
                    conc_emb_anchor = anchors[anchor]
                    conc_emb_element = xs[element]
                    euclidean_similarity = euclidean(conc_emb_anchor, conc_emb_element)
                    similarity_array[anchor] = euclidean_similarity

                relative_representation[element] = similarity_array

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
            Prototypes with the columns: ['PrecursorID', 'Charge', 'Embedding', 'Proteins', 'Sequence', 'Species']
        """

        if verbose:
            print("computing prototypes...")

        support_embeddings = self.get_latent_representations(msms[self._features])

        prototypes = []  # list of (precursor_id, charge, embedding)
        precursor_groups = msms.groupby(["PrecursorID"])
        prototypes_for_plm = []  # list of (proteins, sequence, species) for the PLM

        # exclude prototypes that are responsible for the most false transfers with confidence >= 0.2:
        problematic_prototypes_conf_02 = [
            'Ecoli_LPYITFPEGSEEHTYLHAQR_4',
            'Yeast_LSLTGGFSHHHATDDVEDAAPETK_4',
            'Ecoli_ILEVLQEPDNHHVSAEDLYK_4',
            'Ecoli_DVSLLHKPTTQISDFHVATR_4',
            'Yeast_FGLPHADDVLGLPIGQHIVIK_4',
            'Yeast_TLLEAIDAIEQPSRPTDKPLR_4']

        problematic_prototypes_conf_1 = [
            'Yeast_LSLTGGFSHHHATDDVEDAAPETK_4',
            'Yeast_YAGEVSHDDK_2',
            'Yeast_ITLDTDKLPPHTQIFQAGTK_3',
            'Yeast_EVETEKEEVKEDDSK_3',
            'Yeast_QTVIIAHYPPTVQAGEATK_3',
            'Yeast_FELSGIPPAPR_2',
            'Yeast_AAVDCECEFQNLEHNEK_3',
            'Yeast_GDAGHPSIATTHNHSTSK_3',
            'Ecoli_ILEVLQEPDNHHVSAEDLYK_4',
            'Ecoli_LPYITFPEGSEEHTYLHAQR_4',
            'Yeast_SIQEHPDKNPNDPTATER_3',
            'Yeast_AADTPETSDAVHTEQKPEEEK_3',
            'Yeast_STSGNTHLGGQDFDTNLLEHFK_4'
        ]

        most_problematic_prototypes_conf_1 = [
            'Yeast_LSLTGGFSHHHATDDVEDAAPETK_4'
        ]

        # Dont exclude any prototypes
        for group in precursor_groups:
            (precursor_id,) = group[0]
            #if (precursor_id not in most_problematic_prototypes_conf_1):
            locs = group[1].index
            charge = msms.loc[locs[0], "Charge"]
            ilocs = msms.index.get_indexer(locs)
            proteins = group[1]['Proteins'].iloc[0]
            sequence = group[1]['Sequence'].iloc[0]
            species = group[1]['Species'].iloc[0]
            prototypes.append(
                (precursor_id, charge, np.mean(support_embeddings[ilocs], axis=0), proteins, sequence, species)
            )
            prototypes_for_plm.append((proteins, sequence, species))

        # self._esm_call_list(prototypes_for_plm)

        precursor_ids, charges, embeddings, proteins, sequences, species = zip(*prototypes)

        precursor_ids = np.array(precursor_ids)
        charges = np.array(charges)
        embeddings = np.array(embeddings)
        proteins = np.array(proteins)
        sequences = np.array(sequences)
        species = np.array(species)

        prototypes = {
            "PrecursorID": precursor_ids,
            "Charge": charges,
            "Embedding": embeddings,
            "Protein": proteins,
            "Sequence": sequences,
            "Species": species
        }

        '''
        Get ESM-2 embedding (e.g. 320- or 1280-dimensional). You could use precomputed embeddings or have them computed.
        '''
        esm_embeddings_available = True
        if esm_embeddings_available:
            esm_embeddings_np = np.load('example_data/esm2_t6_8M_UR50D_embeddings.npy')
        else:
            '''
            Prepare ESM:
            models: esm2_t6_8M_UR50D (small), esm2_t33_650M_UR50D (medium), esm2_t36_3B_UR50D (large, 6GB), esm2_t48_15B_UR50D (30GB)
            The next lines could be included in the function get_esm_embedding for clarity
            '''
            model_name = 'esm2_t33_650M_UR50D'
            model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
            batch_converter = alphabet.get_batch_converter()
            model.eval()  # disables dropout for deterministic results
            # work with numpy arrays instead of lists?
            esm_input = list(zip(prototypes['PrecursorID'], prototypes['Sequence']))

            esm_embeddings = self._get_esm_embedding(batch_converter, model, alphabet, esm_input)
            esm_embeddings_np = np.array(esm_embeddings)
            np.save('example_data/{}_embeddings'.format(model_name), esm_embeddings_np)

        prototypes['Esm_embedding'] = esm_embeddings_np

        # save prototypes to file
        np.savez('example_data/all_prototypes.npz', **prototypes)

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
    Function that takes a list of tuples ('PrecursorID', 'Sequence', 'Species') and plots several UMAP embeddings.
    '''
    def _esm_call_list(self,
                       data_with_species: list[tuple[str, str, str]],
                       model_location="esm2_t6_8M_UR50D"
                       ):

        # Load ESM-2 model. It is currently predefined to download "esm2_t6_8M_UR50D".
        # all ESM-2 models: esm2_t6_8M_UR50D, esm2_t33_650M_UR50D, esm2_t36_3B_UR50D (6GB), esm2_t48_15B_UR50D (30GB)
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_location)
        batch_converter = alphabet.get_batch_converter()
        model.eval()  # disables dropout for deterministic results

        # Take only a subset of the data (duplicates allowed) due to extensive computation.
        subset_size = 1000
        random_subset_with_species = random.choices(data_with_species, k=subset_size)

        # Create copy with only the 'Name' and 'Sequence' columns and leave the 'Species' column out.
        random_subset_without_species = [(x[0], x[1]) for x in random_subset_with_species]

        sequence_representations = self._get_esm_embedding(batch_converter, model, alphabet, random_subset_without_species)

        # Reduce ESM-2 representation to two-dimensional UMAP-embedding.
        reducer = umap.UMAP(metric='cosine')
        umap_embedding = reducer.fit_transform(sequence_representations)

        # Plot general UMAP-embedding without species.
        fig, ax = plt.subplots(figsize=(14, 12), ncols=1, nrows=1)
        sp = ax.scatter(umap_embedding[:, 0], umap_embedding[:, 1], s=0.1)
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_title('PLM embedding')
        fig.colorbar(sp)
        plt.show()

        # Connect general UMAP-embedding with species label.
        # Todo: make sure that the order is preserved.
        umap_with_species = []
        for i in range(len(umap_embedding)):
            umap_with_species.append((umap_embedding[i], random_subset_with_species[i][2]))

        # Prepare plot of general UMAP-embedding with species.
        colors = ['red', 'blue', 'green']
        color_map = {species: color for species, color in zip(('HeLa', 'Yeast', 'Ecoli'), colors)}
        species_size = {'HeLa': .1, 'Yeast': .4, 'Ecoli': 1.4}
        fig, ax = plt.subplots(figsize=(14, 14))
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_title('PLM prototypes | latent space - Species')

        # Plot general UMAP-embedding with species.
        for species, color in color_map.items():
            x_data = [x[0] for x in umap_with_species if x[1] == species]
            if len(x_data) > 0:
                umap1, umap2 = zip(*x_data)
                ax.scatter(umap1, umap2, c=color, s=species_size[species], label=species, alpha=0.6)

        ax.legend()
        plt.show()

    '''
    Function that was used to store false/true transfer example prototypes and their neighbors.
    The actual false/true transfer experiments are now conducted in the following two scripts:
    - examples/ESM2_neighbor_transfer_example.py
    - examples/protT5_neighbor_transfer_example_and_embeddings.py
    '''
    def _esm_call_examples(self, model_location, false_positives, true_positives, false_transfer_prototypes, true_transfer_prototypes):

        # Create list of tuples with example prototype sequence and list of (filtered) neighbour sequences
        # Todo: could be simplified by just passing arguments with both PrecursorID and Sequence
        final_false_list = []
        for index, neighbour_list in enumerate(false_transfer_prototypes):
            neighbour_final_list = []
            for neighbour_precursorID in neighbour_list:
                pred_list = neighbour_precursorID.split('_')
                neighbour_sequence = pred_list[1]
                sequence_compatible = True
                if any(char.islower() for char in neighbour_sequence):
                    sequence_compatible = False
                if '(' in neighbour_sequence or ')' in neighbour_sequence:
                    sequence_compatible = False
                if sequence_compatible:
                    neighbour_final_list.append((neighbour_precursorID, neighbour_sequence))
            final_false_list.append((false_positives[index], neighbour_final_list))

        final_true_list = []
        for index, neighbour_list in enumerate(true_transfer_prototypes):
            neighbour_final_list = []
            for neighbour_precursorID in neighbour_list:
                pred_list = neighbour_precursorID.split('_')
                neighbour_sequence = pred_list[1]
                sequence_compatible = True
                if any(char.islower() for char in neighbour_sequence):
                    sequence_compatible = False
                if '(' in neighbour_sequence or ')' in neighbour_sequence:
                    sequence_compatible = False
                if sequence_compatible:
                    neighbour_final_list.append((neighbour_precursorID, neighbour_sequence))
            final_true_list.append((true_positives[index], neighbour_final_list))

        # Save lists to proceed with saved data in another run
        with open('example_data/false_transfer_prototypes_list.pkl', 'wb') as file:
            pickle.dump(false_transfer_prototypes, file)
        with open('example_data/true_transfer_prototypes_list.pkl', 'wb') as file:
            pickle.dump(true_transfer_prototypes, file)
        with open('example_data/final_false_list.pkl', 'wb') as file:
            pickle.dump(final_false_list, file)
        with open('example_data/final_true_list.pkl', 'wb') as file:
            pickle.dump(final_true_list, file)

        # Process is continued in the script transfer_example_neighbors.py


    '''
    Basic function to call for ESM-2 embeddings after setting up a model, alphabet and batch_converter. 
    '''
    def _get_esm_embedding(self, batch_converter, model, alphabet, data):

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
            # Todo: try different layer numbers and fit them to the input model
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[6], return_contacts=True)
            token_representations = results["representations"][6]

            # Generate per-sequence representations via averaging as done in ESM-2 example.
            # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
            for i, tokens_len in enumerate(batch_lens):
                sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))

        return sequence_representations

