import pandas as pd
import numpy as np
from pynndescent import NNDescent


def link_MSMS_to_MS(
    ms2: pd.DataFrame, ms1: pd.DataFrame, verbose=False
) -> pd.DataFrame:
    """
    MaxQuant MBR data remapping.

    Computes the MS1 entry that is closest to each MS2 entry based on
    [Raw file, Charge, m/z, Mass, Intensity, Retention time]

    Afterwards, re-indexes MS2 data to match MS1 datas indices.

    Parameters
    ----------
    ms2 : pd.DataFrame
        MS2 data.

    ms1 : pd.DataFrame
        MS1 data.

    Returns
    -------
    pd.DataFrame
        MS2 data with re-indexed indices.
    """

    ms1 = ms1.reset_index(drop=True)
    ms2 = ms2.reset_index(drop=True)

    features = ["Raw file", "Charge", "m/z", "Mass", "Intensity", "Retention time"]

    ms1_view = ms1[features]
    ms2_view = ms2[features]

    mappings = []

    runs = ms2["Raw file"].unique()

    for i, run in enumerate(runs):
        if verbose:
            print(f"Run {i+1}/{len(runs)}")

        MS1_ref = ms1_view[ms1_view["Raw file"] == run].drop(columns=["Raw file"])
        MS2_ref = ms2_view[ms2_view["Raw file"] == run].drop(columns=["Raw file"])

        knn_index = NNDescent(MS1_ref, metric="euclidean", n_jobs=-1)
        neighbours, _ = knn_index.query(MS2_ref, k=1)

        mappings.append((MS2_ref.index, MS1_ref.iloc[neighbours[:, 0]].index))

    ms2_indices, ms1_indices = zip(*mappings)
    ms2_indices = np.array(np.concatenate(ms2_indices))
    ms1_indices = np.array(np.concatenate(ms1_indices))

    ms2_linked = ms2.iloc[ms2_indices].set_index(ms1_indices)
    return ms2_linked
