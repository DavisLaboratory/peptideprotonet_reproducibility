import pipp
import pipp.utils
import pickle
from argparse import ArgumentParser
from itertools import product
from typing import Dict
import numpy as np
from typing import cast
import pandas as pd
from typing import Any
import os


def _prepare_for_fdr(df_ms: pd.DataFrame) -> pd.DataFrame:
    pos_identifications = df_ms["pred_species"] == "HeLa"
    neg_identifications = ~pos_identifications
    run = df_ms["Raw file"]

    return pd.DataFrame(
        {
            "run": run,
            "pos": pos_identifications,
            "neg": neg_identifications,
        }
    )


def _compute_statistics(
    df_ms_query: pd.DataFrame,
    n_prototypes: Dict[str, int],
) -> Dict[str, Any]:
    confidence_thresholds = np.linspace(0, 1, 20 + 1)

    result_list = []
    for threshold in confidence_thresholds:
        df_ms_query_subset = df_ms_query[df_ms_query["pred_confidence"] >= threshold]
        fdr = pipp.utils.compute_pipp_fdr(_prepare_for_fdr(df_ms_query_subset))
        result_list += [(threshold, fdr)]

    results = pd.DataFrame(result_list, columns=["threshold", "FDR"])

    return {
        "thresholds": confidence_thresholds,
        "fdr": results["FDR"].values,
    }


def _type_int_list(s: str) -> list[int]:
    return [int(item) for item in s.split(",")]


def _get_prototype_counts(df_msms_all: pd.DataFrame) -> Dict[str, int]:
    prototypes = df_msms_all["PrecursorID"].unique()
    prototype_species = [precursor.split("_")[0] for precursor in prototypes]
    n_pos_prototypes = sum([species == "HeLa" for species in prototype_species])
    n_neg_prototypes = sum([species != "HeLa" for species in prototype_species])

    return {
        "n_pos_prototypes": n_pos_prototypes,
        "n_neg_prototypes": n_neg_prototypes,
    }


def run_tests(
    n_pcs_opt: list[int],
    n_anchors_opt: list[int],
    k_neighbours: int,
    sequence_embeddings: Dict[str, np.ndarray] | None,
    checkpoint_path: str,
    df_ms: pd.DataFrame,
    df_msms_all: pd.DataFrame,
    n_repeats: int,
    seed: int,
):
    model = pipp.Peptideprotonet.load(checkpoint_path)

    n_prototypes = _get_prototype_counts(df_msms_all)

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
                df_msms_all,
                k_neighbours=k_neighbours,
                n_anchors=n_anchors,
                ms2_embeddings=ms2_embeddings,
            )

            df_ms["pred_ident"] = identities
            df_ms["pred_confidence"] = confidence
            df_ms["pred_species"] = df_ms["pred_ident"].str.split("_").str[0]

            stats = _compute_statistics(df_ms, n_prototypes)

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
        "--ms_hela_path",
        type=str,
        help="Path to the MS hela data",
        default="data/PXD014777/processed/hela_ms1.csv",
    )

    parser.add_argument(
        "--msms_hela_path",
        type=str,
        help="Path to the MS/MS hela data",
        default="data/PXD014777/processed/hela_ms2.csv",
    )

    parser.add_argument(
        "--msms_species_path",
        type=str,
        help="Path to the MS/MS species data",
        default="data/PXD014777/processed/species_ms2.csv",
    )

    parser.add_argument(
        "--n_repeats",
        type=int,
        help="Number of repeats",
        default=1,
    )

    parser.add_argument("--seed", type=int, help="Random seed", default=42)

    args = parser.parse_args()

    # Read sequence embeddings
    sequence_embeddings = None
    if args.sequence_embedding_path:
        with open(args.sequence_embedding_path, "rb") as f:
            sequence_embeddings = cast(Dict[str, np.ndarray], pickle.load(f))

    df_ms_hela = pd.read_csv(args.ms_hela_path, low_memory=False)
    df_msms_hela = pd.read_csv(args.msms_hela_path, low_memory=False)
    df_msms_species = pd.read_csv(args.msms_species_path, low_memory=False)

    df_msms_all = pd.concat([df_msms_hela, df_msms_species], ignore_index=True)

    # Filter out MS entries that are in MS/MS data
    df_ms_hela.set_index("id", inplace=True, drop=False)
    df_msms_hela.set_index("id", inplace=True, drop=False)
    ms_only_locs = df_ms_hela.index.difference(df_msms_hela.index)
    df_ms_query = df_ms_hela.loc[ms_only_locs]

    # Prefix PrecursorID with Species
    species_precursor_id = df_msms_all["Species"].str.cat(
        df_msms_all["PrecursorID"].astype(str), sep=""
    )
    df_msms_all["PrecursorID"] = species_precursor_id

    # Prefix PrecursorID with Species for sequence embeddings
    if sequence_embeddings is not None:
        species = df_msms_all["Species"].unique()
        sequence_embeddings = {
            f"{specie}{k}": v
            for k, v in sequence_embeddings.items()
            for specie in species
        }

    run_tests(
        args.n_pcs,
        args.n_anchors,
        args.k,
        sequence_embeddings,
        args.checkpoint_path,
        df_ms_query,
        df_msms_all,
        args.n_repeats,
        args.seed,
    )
