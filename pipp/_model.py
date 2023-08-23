import logging
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._module import Encoder

from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)


class Peptideprotonet:

    def __init__(self, device:str=None, dir_path:str=None):
        super().__init__()
        
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        self.device = torch.device(device)

        self.module = Encoder()

        if dir_path is not None:
            self.module.load_state_dict(torch.load(dir_path, map_location=torch.device('cpu')))

        self.module.to(device)
        self.module.eval()


    # @NOTE: this loading method will be inherited from scvi-tools later
    @classmethod
    def load(clz, dir_path:str, device:str=None) -> 'Peptideprotonet':
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
        >>> pipp = pipp.Peptideprotonet.load('path/to/model.pt')
        """

        return Peptideprotonet(device=device, dir_path=dir_path)

    def get_latent_representations(self, x:pd.DataFrame) -> np.ndarray:
        """
        Get the latent representation of the data.

        Parameters
        ----------
        x
            Dataframe with the columns:  ['Charge','Mass', 'm/z', 'Retention time','Retention length', 'Ion mobility index', 'Ion mobility length','Number of isotopic peaks']

        Returns
        -------
            Embeddings of the data.
        """
        
        # assume x is data[attr_names]
        # also could enhance by splitting the data in smaller batches
        # but not required here as observation space small/manageable
        
        x = StandardScaler().fit_transform(x.to_numpy())
        x = torch.from_numpy(x).float().to(self.device)
        z = self.module(x)
        
        latent = z.cpu().detach().numpy()

        return latent