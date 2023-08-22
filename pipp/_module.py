import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self, input_dim=8, hidden_dim=64, latent_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU()
            )

        self.mean_encoder = nn.Linear(hidden_dim, latent_dim)
        self.var_encoder = nn.Linear(hidden_dim, latent_dim) # @NOTE: unused


    def forward(self, x):
        # Simple forward
        hidden = self.encoder(x)
        mu = self.mean_encoder(hidden)
        
        logvar = self.var_encoder(hidden) # @NOTE: unused
        std = logvar                      # @NOTE: unused
        eps = torch.randn_like(std)       # @NOTE: unused
        
        x_sample = mu
        return x_sample