import pandas as pd
import numpy as np
from pynndescent import NNDescent

def link_MSMS_to_MS(MSMS:pd.DataFrame, MS:pd.DataFrame, verbose=False) -> pd.DataFrame:
    """
        MaxQuant MBR data remapping.

        Computes the MS1 entry that is closest to each MS2 entry based on
        [Raw file, Charge, m/z, Mass, Intensity, Retention time]
        
        Afterwards, re-indexes MS2 data to match MS1 datas indices.

        Parameters
        ----------
        MS2 : pd.DataFrame
            MS2 data.
        
        MS1 : pd.DataFrame
            MS1 data.

        Returns
        -------
        pd.DataFrame
            MS2 data with re-indexed indices.
    """

    MS = MS.reset_index(drop=True)
    MSMS = MSMS.reset_index(drop=True)

    features = ['Raw file', 'Charge', 'm/z', 'Mass', 'Intensity', 'Retention time']

    MS1_view = MS[features]
    MS2_view = MSMS[features]

    mappings = []

    runs = MSMS['Raw file'].unique()

    for i, run in enumerate(runs):
        
        if verbose:
            print(f'Run {i+1}/{len(runs)}')
        
        MS1_ref = MS1_view[MS1_view['Raw file'] == run].drop(columns=['Raw file'])
        MS2_ref = MS2_view[MS2_view['Raw file'] == run].drop(columns=['Raw file'])

        knn_index = NNDescent(MS1_ref, metric='euclidean', n_jobs=-1)
        neighbours, _ = knn_index.query(MS2_ref, k=1)

        mappings.append((MS2_ref.index, MS1_ref.iloc[neighbours[:,0]].index))

    MS2_indices, MS1_indices = zip(*mappings)
    MS2_indices = np.array(np.concatenate(MS2_indices))
    MS1_indices = np.array(np.concatenate(MS1_indices))

    MS2_linked = MSMS.iloc[MS2_indices].set_index(MS1_indices)
    return MS2_linked
