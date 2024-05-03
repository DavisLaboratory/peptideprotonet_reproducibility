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
import scanpy
from sklearn.decomposition import PCA

from ._module import Encoder

logger = logging.getLogger(__name__)

'''
Files to generate:
- example_data/esm2_t33_650M_UR50D_embeddings.npy
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
        esm2_model="esm2_t33_650M_UR50D",
        use_precomputed_esm_embeddings=False,
        esm_embedding_path='example_data/esm2_t33_650M_UR50D_embeddings_6.npy',
        n_anchors=300,
        n_principal_components=10,
        n_representation_layer=6,
        generate_visualizations=True,
        conduct_neighbour_experiment=False,
        use_preselected_anchors=False,
        anchors_idx=[],
        use_concatenated_embeddings=True,
        dataset="HeLa_Yeast_Ecoli"
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

        conduct_neighbour_experiment
            Whether to generate output files that are necessary to conduct the species separation experiment
            with false positive and true positive examples and their neighbours.

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
            use_precomputed_esm_embeddings=use_precomputed_esm_embeddings,
            esm_embedding_path=esm_embedding_path,
            n_representation_layer=n_representation_layer,
            conduct_neighbour_experiment=conduct_neighbour_experiment,
            dataset=dataset)

        identities, confidence = self._propagate_using_prototypes(
            ms,
            prototypes,
            k_neighbours=k_neighbours,
            use_anchors=use_anchors,
            verbose=verbose,
            n_anchors=n_anchors,
            n_principal_components=n_principal_components,
            generate_visualizations=generate_visualizations,
            conduct_neighbour_experiment=conduct_neighbour_experiment,
            use_preselected_anchors=use_preselected_anchors,
            anchors_idx=anchors_idx,
            use_concatenated_embeddings=use_concatenated_embeddings
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
        generate_visualizations=False,
        conduct_neighbour_experiment=False,
        use_preselected_anchors=False,
        anchors_idx=[],
        use_concatenated_embeddings=True
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

        generate_visualizations
            Whether to generate visualizations.

        conduct_neighbour_experiment
            Whether to save true und false transfer examples and their neighbours.

        Returns
        -------
            Predicted identities and confidence.
        """

        if conduct_neighbour_experiment:
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

        # Get MS1 embedding of query precursors
        query_embeddings = self.get_latent_representations(ms[self._features])
        query_charges = ms["Charge"].values

        print('Computing relative representations...')

        if use_anchors:
            means = np.mean(query_embeddings, axis=0)
            query_embeddings -= means
            prototypes["MS1_Embedding"] -= means

            if use_concatenated_embeddings:

                # Retrieve the first n principal components of the ESM-2 embeddings.
                principal_components_esm = scanpy.pp.pca(prototypes['ESM_Embedding'], n_comps=n_principal_components)
                prototypes['Concatenated_Embedding'] = np.hstack((prototypes['MS1_Embedding'], principal_components_esm))

                if generate_visualizations:

                    # Compute Principal Component Analysis of ESM-2 Embeddings for visualization.
                    # Here, another library is used for PCA than before, simply because of convenience of the offered functions.
                    pca = PCA()
                    pca.fit(prototypes['ESM_Embedding'])

                    # Plot Cumulative Explained Variance.
                    plt.figure(figsize=(10, 8))
                    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_),
                             marker='o')
                    plt.axvline(x=25, color='black', linestyle='--')
                    plt.xlabel('Principal Component Number', fontsize=25)
                    plt.ylabel('Cumulative Explained Variance', fontsize=25)
                    plt.ylim(0, 1.05)
                    plt.title('Cumulative Explained Variance', fontsize=25)
                    plt.xticks([30,200,400,600,800,1000,1200], fontsize=20)
                    #plt.xscale('log')
                    plt.yticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],fontsize=20)
                    plt.show()

                    # Scree plot.
                    plt.figure(figsize=(10, 8))
                    plt.plot(pca.explained_variance_ratio_,
                             marker='o')
                    plt.xticks([30,200,400,600,800,1000,1200], fontsize=20)
                    plt.yticks(fontsize=20)
                    plt.axvline(x=25, color='black', linestyle='--')
                    plt.xlabel('Principal Component Number', fontsize=25)
                    plt.ylabel('Individual Explained Variance', fontsize=25)
                    #plt.xscale('log')
                    plt.title('Scree Plot', fontsize=25)
                    plt.show()

                    # Plot UMAP of concatenated embeddings colored by charge.
                    reducer = umap.UMAP(metric='cosine')
                    umap_embedding = reducer.fit_transform(prototypes['Concatenated_Embedding'])
                    fig, ax = plt.subplots(figsize=(14, 14))
                    sp = ax.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=prototypes["Charge"], s=0.1)
                    ax.set_xlabel('UMAP1')
                    ax.set_ylabel('UMAP2')
                    ax.set_title('Prototype conc. emb. | latent space - Charge')
                    fig.colorbar(sp)
                    plt.show()

                '''
                1. Filters out prototypes that are not from the species 'HeLa'. 
                   Should be generalized on the species from the query data set in the future.
                2. Either:
                    a. Selects a random subset of the filtered prototypes as anchors.
                    b. Selects the prototypes of which the indexes were already specified in 'anchors_idx'.
                3. Returns anchor prototypes.
                '''
                anchors = self._select_anchors(
                    prototypes,
                    n_anchors=n_anchors,
                    use_preselected_anchors=use_preselected_anchors,
                    preselected_anchors_idx=anchors_idx,
                    use_concatenated_embeddings=use_concatenated_embeddings)

                # Compute Relative Representations of Prototypes based on Concatenated MS1-ESM Embeddings
                prototype_representation = self._compute_relative_representations(
                    prototypes['Concatenated_Embedding'], anchors['Concatenated_Embedding']
                )

                # Compute Relative Representations of Query Precursors based on MS1 Embeddings
                query_representation = self._compute_relative_representations(
                    query_embeddings, anchors['MS1_Embedding']
                )

            else:
                anchors = self._select_anchors(
                    prototypes,
                    n_anchors=n_anchors,
                    use_preselected_anchors=use_preselected_anchors,
                    preselected_anchors_idx=anchors_idx,
                    use_concatenated_embeddings=use_concatenated_embeddings)

                # Compute Relative Representations of Prototypes based on MS1 Embeddings
                prototype_representation = self._compute_relative_representations(
                    prototypes['MS1_Embedding'], anchors['MS1_Embedding']
                )

                # Compute Relative Representations of Query Precursors based on MS1 Embeddings
                query_representation = self._compute_relative_representations(
                    query_embeddings, anchors['MS1_Embedding']
                )

            if generate_visualizations:
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

                # plot different features on the UMAP of the relative representation of the concatenated embedding
                fig, ax = plt.subplots(figsize=(14, 14))
                sp = ax.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=prototypes['Charge'], s=0.1)
                ax.set_xlabel('UMAP1')
                ax.set_ylabel('UMAP2')
                ax.set_title('Prototype rel. repr. conc. emb. | Charge')
                fig.colorbar(sp)
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

                # Plot different features on the UMAP of the MS1 relative representation.
                fig, axs = plt.subplots(figsize=(14, 12), ncols=3, nrows=2)

                ax = axs[0][0]
                sp = ax.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=ms['Retention time'], s=0.1)
                ax.set_xlabel('UMAP1')
                ax.set_ylabel('UMAP2')
                ax.set_title('MS1 relative representation | latent space - RT')
                fig.colorbar(sp)

                ax = axs[0][1]
                sp = ax.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=ms['Charge'], s=0.1)
                ax.set_xlabel('UMAP1')
                ax.set_ylabel('UMAP2')
                ax.set_title('MS1 relative representation | latent space - Charge')
                fig.colorbar(sp)

                ax = axs[1][0]
                sp = ax.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=ms['Mass'], s=0.1)
                ax.set_xlabel('UMAP1')
                ax.set_ylabel('UMAP2')
                ax.set_title('MS1 relative representation | latent space - Mass')
                fig.colorbar(sp)

                ax = axs[1][1]
                sp = ax.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=ms['m/z'], s=0.1)
                ax.set_xlabel('UMAP1')
                ax.set_ylabel('UMAP2')
                ax.set_title('MS1 relative representation | latent space - m/z')
                fig.colorbar(sp)

                ax = axs[1][2]
                sp = ax.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=ms['Number of isotopic peaks'],
                                s=0.1)
                ax.set_xlabel('UMAP1')
                ax.set_ylabel('UMAP2')
                ax.set_title('MS1 relative representation | latent space - Number of isotopic peaks')
                fig.colorbar(sp)
                plt.show()

            # recover original embeddings - keep in-place to save memory
            query_embeddings += means
            prototypes['MS1_Embedding'] += means

        else:
            # Without Relative Representations: MS1 Embedding is the representation. ESM-2 embeddings are not used here.
            prototype_representation = prototypes['MS1_Embedding']
            query_representation = query_embeddings

        print("setting up nearest neighbour graph...")

        knn_index = NNDescent(
            prototype_representation, metric=distance_metric
        )

        if verbose:
            print(f"computing {k_neighbours}-nearest neighbour prototypes of query precursors...")


        if conduct_neighbour_experiment:
            # Get neighbour indexes of true/false transfer prototype examples
            false_neighbours, false_distances = knn_index.query(prototype_representation[false_positive_indexes], k=5)
            true_neighbours, true_distances = knn_index.query(prototype_representation[true_positive_indexes], k=5)

            # Get neighbour PrecursorIDs
            false_neighbour_ids = prototypes['PrecursorID'][false_neighbours]
            true_neighbour_ids = prototypes['PrecursorID'][true_neighbours]

            # Get ESM-2 embeddings of example prototypes and their neighbours.
            self._store_false_true_positives_and_neighbours("esm2_t33_650M_UR50D", false_positives, true_positives,
                                                            false_neighbour_ids, true_neighbour_ids)

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
        use_concatenated_embeddings=True
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

        '''
        TODO: Select anchors only from peptides that were present in 9/10 or 10/10 runs. Does not work yet.
        '''
        preselect_anchors = False
        if preselect_anchors:
            # filter out peptides that were in too few runs
            filtered_latent_embeddings = prototypes['MS1_Embedding'][
                ((prototypes['Run_Count'] > 5) & ((prototypes['Species'] == 'Ecoli') | (prototypes['Species'] == 'Yeast'))) |
                ((prototypes['Run_Count'] > 15) & (prototypes['Species'] == 'HeLa'))]
            filtered_concatenated_embeddings = prototypes['Concatenated_Embedding'][
                ((prototypes['Run_Count'] > 5) & ((prototypes['Species'] == 'Ecoli') | (prototypes['Species'] == 'Yeast'))) |
                ((prototypes['Run_Count'] > 15) & (prototypes['Species'] == 'HeLa'))]

            anchors_idx = np.random.choice(
                filtered_latent_embeddings.shape[0], size=n_anchors, replace=False
            )
            anchors = {'MS1_embedding': filtered_latent_embeddings[anchors_idx],
                        'Concatenated_embedding': filtered_concatenated_embeddings[anchors_idx]}

        else:
            # prefilter prototypes for species from query data set (here: HeLa)
            right_species_indices = np.where(prototypes['Species'] == 'HeLa')[0]
            filtered_prototypes = {key: value[right_species_indices] for key, value in prototypes.items()}

            if use_preselected_anchors:

                # Take anchors from preselected_anchors_idx.
                if use_concatenated_embeddings:
                    # Store coupled (MS1, MS1+ESM) anchors.
                    anchors = {'MS1_Embedding': filtered_prototypes['MS1_Embedding'][preselected_anchors_idx],
                               'Concatenated_Embedding': filtered_prototypes['Concatenated_Embedding'][preselected_anchors_idx]}
                else:
                    # Store simple (MS1) anchors. Here, the ESM-embedding is not used.
                    anchors = {'MS1_Embedding': filtered_prototypes['MS1_Embedding'][preselected_anchors_idx]}
            else:

                # Select a set of random anchor points.
                random_state = np.random.RandomState()
                anchors_idx = random_state.choice(
                    filtered_prototypes['MS1_Embedding'].shape[0], size=n_anchors, replace=False
                )

                if use_concatenated_embeddings:
                    # Store coupled (MS1, MS1+ESM) anchors
                    anchors = {'MS1_Embedding': filtered_prototypes['MS1_Embedding'][anchors_idx],
                               'Concatenated_Embedding': filtered_prototypes['Concatenated_Embedding'][anchors_idx]}
                else:
                    # Store simple (MS1) anchors. Here, the ESM-embedding is not used.
                    anchors = {'MS1_Embedding': filtered_prototypes['MS1_Embedding'][preselected_anchors_idx]}

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
        verbose=False,
        use_precomputed_esm_embeddings=False,
        esm2_model="esm2_t33_650M_UR50D",
        esm_embedding_path='example_data/embeddings_esm2_t6_8M_UR50D_layer6_datasetHeLa_Yeast_Ecoli.npy',
        n_representation_layer=6,
        conduct_neighbour_experiment=False,
        dataset="HeLa_Yeast_Ecoli"
    ) -> Dict[str, np.ndarray]:
        """
        Helper function to compute prototypes.

        Parameters
        ----------
        msms
            Support set with the columns: ['PrecursorID', 'Charge', 'Mass', 'm/z', 'Retention time', 'Retention length', 'Ion mobility index', 'Ion mobility length', 'Number of isotopic peaks']

        verbose
            Whether to print progress.

        use_precomputed_esm_embeddings
            Whether to use pre-computed ESM-2 embeddings. Path needs to be specified in 'esm_embedding_path'.

        esm_embedding_path
            Relative path to the pre-computed ESM-2 embeddings.

        n_representation_layer
            The n-th ESM-2 layer of which to retrieve the embeddings.

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
            proteins = group[1]['Proteins'].iloc[0]
            sequence = group[1]['Sequence'].iloc[0]
            if 'Species' in group[1].columns:
                species = group[1]['Species'].iloc[0]
            else: # only applicable for the HeLa example notebook
                species = 'HeLa'
            run_count = len(group[1])
            prototypes.append(
                (precursor_id, charge, np.mean(support_embeddings[ilocs], axis=0), proteins, sequence, species, run_count)
            )

        precursor_ids, charges, ms1_embeddings, proteins, sequences, species, run_count = zip(*prototypes)

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
            "Run_Count": run_count
        }

        '''
        Get ESM-2 embedding (e.g. 320- or 1280-dimensional). You could use precomputed embeddings or have them computed.
        The files with the '_H' at the end are for the HeLa example notebook
        '''
        if use_precomputed_esm_embeddings:
            esm_embeddings_np = np.load(esm_embedding_path)
        else:

            # sets up model and alphabet of specified esm2_model.
            model, alphabet = esm.pretrained.load_model_and_alphabet(esm2_model)

            # is necessary for ESM to process batches of input sequences.
            batch_converter = alphabet.get_batch_converter()

            # disables dropout for deterministic results.
            model.eval()

            # prepare input for ESM as list of tuples (PrecursorID, Sequence).
            esm_input = list(zip(prototypes['PrecursorID'], prototypes['Sequence']))

            # compute ESM embeddings
            esm_embeddings = self._get_esm_embedding(batch_converter, model, alphabet, esm_input, n_representation_layer)

            # store ESM embeddings as a numpy array.
            esm_embeddings_np = np.array(esm_embeddings)
            np.save(f'example_data/embeddings_{esm2_model}_layer{n_representation_layer}_dataset{dataset}', esm_embeddings_np)

        prototypes['ESM_Embedding'] = esm_embeddings_np

        if conduct_neighbour_experiment:
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
    Function that was used to store false/true transfer example prototypes and their neighbors.
    The actual species separation experiment is conducted in species_separation_experiment.py
    '''
    def _store_false_true_positives_and_neighbours(self, model_location, false_positives, true_positives, false_transfer_prototypes, true_transfer_prototypes):

        # Create list of tuples with example prototype sequence and list of (filtered) neighbour sequences
        # could be simplified by just passing arguments with both PrecursorID and Sequence
        false_positives_and_neighbours = []
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
            false_positives_and_neighbours.append((false_positives[index], neighbour_final_list))

        true_positives_and_neighbours = []
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
            true_positives_and_neighbours.append((true_positives[index], neighbour_final_list))

        # Save lists to proceed with saved data in species_separation_experiment.py
        with open('example_data/false_positives_and_neighbours.pkl', 'wb') as file:
            pickle.dump(false_positives_and_neighbours, file)
        with open('example_data/true_positives_and_neighbours.pkl', 'wb') as file:
            pickle.dump(true_positives_and_neighbours, file)

        # Process is continued in the script species_separation_experiment.py


    def _get_esm_embedding(
            self,
            batch_converter,
            model,
            alphabet,
            data,
            n_representation_layer):
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

        n_representation_layer
            The n-th ESM-2 layer from which to retrieve the representations.

        Returns
        -------
            List of representations.

        """

        sequence_representations = []

        # Iterate through the list in bulks because the execution is more stable that way.
        # bulk_size of 500 is not necessarily the best size.
        bulk_size = 500
        for i in range(0, len(data), bulk_size):
            print("ESM-2 embeddings of peptide sequences already computed:", i)
            bulk = data[i:i + bulk_size]

            # Use batch converter as recommended by ESM-2 documentation.
            batch_labels, batch_strs, batch_tokens = batch_converter(bulk)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

            # Extract per-residue representations (on CPU) as recommended by ESM-2 documentation.
            # n_representation_layer specifies the ESM-2 layer from which to take the representation.
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[n_representation_layer], return_contacts=True)
            token_representations = results["representations"][n_representation_layer]

            # Generate per-sequence representations via averaging as recommended by ESM-2 documentation.
            # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
            for i, tokens_len in enumerate(batch_lens):
                sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))

        return sequence_representations

