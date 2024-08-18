import pandas as pd
import numpy as np
from pynndescent import NNDescent
import scanpy as sc
import os


def link_MSMS_to_MS(
    msms: pd.DataFrame, ms: pd.DataFrame, verbose=False
) -> pd.DataFrame:
    """
    MaxQuant MBR data remapping.

    Computes the MS entry that is closest to each MS/MS entry based on
    [Raw file, Charge, m/z, Mass, Intensity, Retention time]

    Afterwards, re-indexes MS/MS data to match MS datas indices.

    Parameters
    ----------
    msms : pd.DataFrame
        MS/MS data.

    ms : pd.DataFrame
        MS data.

    Returns
    -------
    pd.DataFrame
        MS/MS data with re-indexed indices.
    """
    assert set(ms["Raw file"].unique()) == set(msms["Raw file"].unique())
    assert "id" in ms.columns

    features = ["Charge", "m/z", "Mass", "Intensity", "Retention time"]

    ms_ids_all = []
    msms_loc_all = []

    runs = msms["Raw file"].unique()

    for i, run in enumerate(runs):
        if verbose:
            print(f"Run {i+1}/{len(runs)}")

        ms_subset = ms[ms["Raw file"] == run]
        msms_subset = msms[msms["Raw file"] == run]

        knn_index = NNDescent(ms_subset[features], metric="euclidean", n_jobs=-1)
        neighbours, _ = knn_index.query(msms_subset[features], k=1)

        msms_loc = msms_subset.index.values  # msms index
        ms_subset_iloc = neighbours[:, 0]  # ms subset index
        ms_id = ms_subset.iloc[ms_subset_iloc]["id"].values  # id

        ms_ids_all += ms_id.tolist()
        msms_loc_all += msms_loc.tolist()

    msms_locs = np.array(msms_loc_all)
    ms_ids = np.array(ms_ids_all)

    msms_match = msms.loc[msms_locs].copy()
    msms_match["id"] = ms_ids

    assert msms_match.shape[0] == msms.shape[0]

    return msms_match


def pca(_dict, n_pcs=10):
    sequence_embeddings = np.array(list(_dict.values()))
    embeddings = sc.pp.pca(sequence_embeddings, n_comps=n_pcs, return_info=False)
    return dict(zip(_dict.keys(), embeddings))


def filter_by_highest_intensity(
    df: pd.DataFrame,
    group_key=["Raw file", "PrecursorID"],
    intensity_key="Intensities",
) -> pd.DataFrame:
    return df.loc[df.groupby(group_key)[intensity_key].idxmax()]


def create_abundance_matrix(
    df: pd.DataFrame,
    obs_key="Raw file",
    var_key="PrecursorID",
    x_key="Intensities",
    layer_keys: list[str] | None = None,
) -> sc.AnnData:
    assert df.groupby([obs_key, var_key]).size().max() == 1, "Duplicated entries"

    x = df.pivot(index=obs_key, columns=var_key, values=x_key)

    layers = {"raw": x.copy()}

    if layer_keys:
        for key in layer_keys:
            df_extra = df.pivot(index=obs_key, columns=var_key, values=key)
            layers[key] = df_extra

    return sc.AnnData(X=x, layers=layers)


def reshape_anndata_like(adata, adata_like, sanity_check=True, verbose=False):
    """
    Reshape adata.X into the shape of adata_like.X.
    Annotations in .obs and .var are copied to the result.
    """

    def intersection(a, b):
        return list(set(a).intersection(b))

    row_overlap = intersection(adata.obs.index, adata_like.obs.index)
    col_overlap = intersection(adata.var.index, adata_like.var.index)

    if verbose:
        print(
            f"transferring {len(row_overlap)} cells and {len(col_overlap)} proteins during reshape."
        )

    x = pd.DataFrame(np.nan, index=adata_like.obs.index, columns=adata_like.var.index)
    x.loc[row_overlap, col_overlap] = adata[row_overlap, col_overlap].X.copy()

    obs = pd.DataFrame(
        np.nan,
        index=adata_like.obs.index,
        columns=adata.obs.columns,
    )
    obs = obs.astype(object)
    obs.loc[row_overlap, :] = adata[row_overlap, :].obs.copy()

    var = pd.DataFrame(
        np.nan,
        index=adata_like.var.index,
        columns=adata.var.columns,
    )
    var = var.astype(object)
    var.loc[col_overlap, :] = adata[:, col_overlap].var.copy()

    result = sc.AnnData(
        x,
        obs=obs,
        var=var,
    )

    if sanity_check:
        assert np.all(result.obs.index == adata_like.obs.index)
        assert np.all(result.var.index == adata_like.var.index)

        assert np.all(result.obs.columns == adata.obs.columns)
        assert np.all(result.var.columns == adata.var.columns)

        a = result[row_overlap, col_overlap].copy()
        a.X[np.isnan(a.X)] = 0

        b = adata[row_overlap, col_overlap].copy()
        b.X[np.isnan(b.X)] = 0

        assert np.all(a.X == b.X)

    return result


def compute_pipp_fdr(df: pd.DataFrame) -> float:
    return sum(df["neg"]) / df.shape[0]


def compute_diann_fdr(
    df: pd.DataFrame,
    n_pos_prototypes: int,
    n_neg_prototypes: int,
) -> float:
    n_pos = sum(df["pos"])
    n_neg = sum(df["neg"])
    return (n_neg / n_pos) * (n_pos_prototypes / n_neg_prototypes)


def compute_diann_avg_fdr(
    df: pd.DataFrame,
    n_pos_prototypes: int,
    n_neg_prototypes: int,
) -> float:
    fdr_by_run = []

    runs = df["run"].unique()
    for run in runs:
        df_run = df[df["run"] == run]
        n_pos = sum(df_run["pos"])
        n_neg = sum(df_run["neg"])
        fdr = (n_neg / n_pos) * (n_pos_prototypes / n_neg_prototypes)
        fdr_by_run.append(fdr)

    return np.mean(fdr_by_run)


def compute_diann_avg_unweighted_fdr(df: pd.DataFrame) -> float:
    fdr_by_run = []

    runs = df["run"].unique()
    for run in runs:
        df_run = df[df["run"] == run]
        n_pos = sum(df_run["pos"])
        n_neg = sum(df_run["neg"])
        fdr = n_neg / n_pos
        fdr_by_run.append(fdr)

    return np.mean(fdr_by_run)


def compute_experimential_fdr(
    df: pd.DataFrame,
    n_pos_prototypes: int,
    n_neg_prototypes: int,
) -> float:
    """Compute the experimential FDR as described in the paper:
    [1] https://www.nature.com/articles/s41467-022-31492-0#Sec7
    """
    n_pos = sum(df["pos"])
    n_neg = sum(df["neg"])

    q = 0  # not relevant
    pi0 = (n_neg_prototypes + n_pos_prototypes - 0.95 * q) / (
        n_neg_prototypes + n_pos_prototypes
    )
    fdr = n_neg / (n_neg + n_pos) * (n_neg_prototypes + n_pos_prototypes) / n_neg * pi0

    return fdr


def flat_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flat_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
