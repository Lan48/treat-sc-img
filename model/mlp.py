import torch
import torch.nn as nn

class MLPEncoder(nn.Module):
    def __init__(self, patch_dim=512, latent_dim=512):
        super().__init__()
        layers = [
            nn.Linear(patch_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, patch_dim),
            nn.LayerNorm(patch_dim)
        ]
        self.mlp = nn.Sequential(*layers)

    def forward(self, embeddings):
        return self.mlp(embeddings)

