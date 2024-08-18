import pipp
import pickle
from argparse import ArgumentParser
from itertools import product
from typing import Dict
import numpy as np
from typing import cast
import pandas as pd
from typing import Any
import scanpy as sc
import os

import pipp.utils


def _compute_statistics(
    ms_results: pd.DataFrame,
    msms_results: pd.DataFrame,
    confidence_threshold=0.5,
) -> Dict[str, Any]:
    # Create AnnData object for MS data
    df_x = pipp.utils.filter_by_highest_intensity(ms_results)
    ms_adata = pipp.utils.create_abundance_matrix(df_x, layer_keys=["confidence"])
    sc.pp.log1p(ms_adata, layer="raw")

    # Create AnnData object for MS/MS data
    df_x = pipp.utils.filter_by_highest_intensity(msms_results)
    msms_adata = pipp.utils.create_abundance_matrix(df_x)
    sc.pp.log1p(msms_adata, layer="raw")

    # Create joint AnnData object
    adata = sc.AnnData(obs=msms_adata.obs, var=msms_adata.var)
    adata.layers["msms"] = msms_adata.layers["raw"].copy()

    ms_adata.X = ms_adata.layers["raw"].copy()
    adata_like = pipp.utils.reshape_anndata_like(ms_adata, adata)
    adata.layers["ms"] = adata_like.X.copy()

    ms_adata.X = ms_adata.layers["confidence"].copy()
    adata_like = pipp.utils.reshape_anndata_like(ms_adata, adata)
    adata.layers["ms_confidence"] = adata_like.X.copy()

    # Add MS layer with confidence threshold
    mask = adata.layers["ms_confidence"] > confidence_threshold
    adata.layers["ms_threshold"] = adata.layers["ms"].copy()
    adata.layers["ms_threshold"][~mask] = np.nan

    nan_mask = np.isnan(adata.layers["msms"])

    adata.layers["combined"] = adata.layers["msms"].copy()
    adata.layers["combined"][nan_mask] = adata.layers["ms_threshold"][nan_mask]

    # Look at the highly confident results
    mask = np.any(~np.isnan(adata.layers["ms_threshold"]), axis=0)
    adata_subset = adata[:, mask]

    adata_subset_t = adata_subset.copy().T

    # Compute NaN proportion
    prop_nan_before = np.mean(np.isnan(adata_subset_t.layers["msms"]))
    prop_nan_after = np.mean(np.isnan(adata_subset_t.layers["combined"]))

    # Compute coefficient of variation
    mean = np.nanmean(adata_subset_t.layers["combined"], axis=1)
    std = np.nanstd(adata_subset_t.layers["combined"], axis=1)
    cv = np.mean(std / mean)

    return {
        "p_nan_before": prop_nan_before,
        "p_nan_after": prop_nan_after,
        "cv": cv,
    }


def _type_int_list(s: str) -> list[int]:
    return [int(item) for item in s.split(",")]


def run_tests(
    n_pcs_opt: list[int],
    n_anchors_opt: list[int],
    k_neighbours: int,
    sequence_embeddings: Dict[str, np.ndarray] | None,
    checkpoint_path: str,
    df_ms: pd.DataFrame,
    df_msms: pd.DataFrame,
    n_repeats: int,
    seed: int,
):
    model = pipp.Peptideprotonet.load(checkpoint_path)

    if sequence_embeddings is None:
        n_pcs_opt = [0]

    for n_pcs, n_anchors in product(n_pcs_opt, n_anchors_opt):
        ms2_embeddings = None
        if (sequence_embeddings is not None) and (n_pcs > 0):
            ms2_embeddings = pipp.utils.pca(sequence_embeddings, n_pcs)

        for repeat in range(n_repeats):
            # set seed here so each "(n_pcs, n_anchor)-test" uses the same set of candidate anchors.
            # And reset the seed each run in order to reproduce the results when e.g., `n_repeats=1`.
            run_seed = seed + repeat
            np.random.seed(run_seed)

            print(
                f"Running test with {n_pcs} PCs, {n_anchors} anchors and seed {run_seed}."
            )

            identities, confidence = model.propagate(
                df_ms,
                df_msms,
                k_neighbours=k_neighbours,
                n_anchors=n_anchors,
                ms2_embeddings=ms2_embeddings,
            )

            df_ms_results = pd.DataFrame(
                {
                    "Raw file": df_ms["Raw file"],
                    "Intensities": df_ms["Intensity"],
                    "PrecursorID": identities,
                    "confidence": confidence,
                }
            )

            df_msms_results = pd.DataFrame(
                {
                    "Raw file": df_msms["Raw file"],
                    "Intensities": df_msms["Intensity"],
                    "PrecursorID": df_msms["PrecursorID"],
                }
            )

            stats = _compute_statistics(df_ms_results, df_msms_results)

            result = {
                "n_pcs": n_pcs,
                "n_anchors": n_anchors,
                "k_neighbours": k_neighbours,
                "seed": run_seed,
                "stats": stats,
            }

            output_dir = "results"
            os.makedirs(output_dir, exist_ok=True)

            filename = f"results_pcs{n_pcs}_anchors{n_anchors}_k{k_neighbours}_seed{run_seed}.pkl"
            output_path = os.path.join(output_dir, filename)

            with open(output_path, "wb") as f:
                pickle.dump(result, f)

    print("Done!")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "n_pcs",
        help="Number of PCs (delimited list input, e.g. '5,10,15')",
        type=_type_int_list,
    )
    parser.add_argument(
        "n_anchors",
        help="Number of anchors (delimited list input, e.g. '10,20,30')",
        type=_type_int_list,
    )

    parser.add_argument(
        "--k",
        type=int,
        help="Number of prototype neighbours to consider for each query",
        default=10,
    )

    parser.add_argument(
        "--sequence_embedding_path",
        type=str,
        help="Path to the sequence embeddings",
        default=None,
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to the PIPPs checkpoint",
        default="checkpoints/checkpoint1.pth",
    )

    parser.add_argument(
        "--ms_path",
        type=str,
        help="Path to the MS data",
        default="data/PXD014777/processed/hela_ms1.csv",
    )

    parser.add_argument(
        "--msms_path",
        type=str,
        help="Path to the MS/MS data",
        default="data/PXD014777/processed/hela_ms2.csv",
    )

    parser.add_argument(
        "--n_repeats",
        type=int,
        help="Number of repeats",
        default=1,
    )

    parser.add_argument("--seed", type=int, help="Random seed", default=42)

    args = parser.parse_args()

    # read files
    sequence_embeddings = None
    if args.sequence_embedding_path:
        with open(args.sequence_embedding_path, "rb") as f:
            sequence_embeddings = cast(Dict[str, np.ndarray], pickle.load(f))

    df_ms_hela = pd.read_csv(args.ms_path, low_memory=False)
    df_msms_hela = pd.read_csv(args.msms_path, low_memory=False)

    df_ms_hela.set_index("id", inplace=True, drop=False)
    df_msms_hela.set_index("id", inplace=True, drop=False)
    ms1_only_locs = df_ms_hela.index.difference(df_msms_hela.index)
    df_ms_query = df_ms_hela.loc[ms1_only_locs]

    run_tests(
        args.n_pcs,
        args.n_anchors,
        args.k,
        sequence_embeddings,
        args.checkpoint_path,
        df_ms_query,
        df_msms_hela,
        args.n_repeats,
        args.seed,
    )
