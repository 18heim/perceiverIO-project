import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math
from icecream import ic


class PositionalEncoding(nn.Module):
    "Position embeddings"
    def __init__(self, d_model: int, max_len: int = 5000):
        """Génère des embeddings de position
        Args:
            d_model (int): Dimension des embeddings à générer
            max_len (int, optional): Longueur maximale des textes.
                Attention, plus cette valeur est haute, moins bons seront les embeddings de position.
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            div_term = torch.exp(torch.arange(0, d_model - 1, 2, dtype=torch.float) *
                                -(math.log(10000.0) / d_model))
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Ajoute les embeddings de position"""
        x = x + self.pe[:, :x.size(1)]
        return x