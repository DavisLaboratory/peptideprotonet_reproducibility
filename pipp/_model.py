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


class peptideprotonet:

    def __init__(self, reference_model:str):
        super().__init__()
        
        # TODO: replaced with scvi-tools's approach for initialization
        
        # load the model and state dict 
        self.model = Encoder() # default params specifief in Encoder are identical as pretrained model - that's why it works
        self.model.load_state_dict(torch.load(reference_model))
        self.model.eval()
        
        # TODO define device
        
        


    def get_latent_representations(self, x):

        # assume x is data[attr_names]
        # also could enhance by splitting the data in smaller batches
        # but not required here as observation space small/manageable
        
#         x = torch.from_numpy(data[attr_names].to_numpy()).float()
        x = torch.from_numpy(x.to_numpy()).float()
        x = StandardScaler().fit_transform(x)
        z = model(torch.from_numpy(x).float().to(device))
        
        # TODO: check scvi-tools re: how this is handeled
        z = z_query.cpu().detach().numpy()
        return z