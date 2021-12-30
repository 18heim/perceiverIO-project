from code.perceiver.perceiver import BasicDecoder, CrossAttention
import torch
from torch import nn
import torch.nn.functional as F
import math

from attention_utils import PerceiverDecoder, LatentTransformerBlock, CrossAttentionBlock

class PerceiverIO(nn.Module):
    #TODO: This is only pseudo code
    def __init__(self,
                 num_latent_heads,
                 num_cross_heads,
                 in_dim,
                 num_latent_block,
                 qlatent_dim,
                 qout_dim,
                 qk_dim,
                 v_dim,
                 out_dim,
                 dropout_prob
                 ) -> None:
        super(PerceiverIO, self).__init__()
        self.encoder_block = CrossAttentionBlock(in_dim, qlatent_dim, qk_dim, v_dim, num_cross_heads, dim_cross_feedforward, dropout_prob)
        self.latent_block = nn.ModuleList([LatentTransformerBlock(in_dim, qk_dim, v_dim, num_latent_heads, dim_latent_feedforward, dropout_prob) for l in range(num_latent_block)])
        self.decoder = PerceiverDecoder(qlatent_dim, qout_dim, out_dim, qk_dim, v_dim, num_cross_heads, dim_cross_feedforward, dropout_prob)

    def forward(self, x):
        latent_q = self.init_latent_q()
        output_q = self.init_out_q()
        # Encoder block
        latent_q = self.encoder_block(x, latent_q)
        # Latent blocks
        latent_q = self.latent_block(latent_q)
        # Decoder blocks
        out = self.decoder(latent_q, output_q)

        return out