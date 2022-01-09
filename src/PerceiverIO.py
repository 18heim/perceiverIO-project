from code.perceiver.perceiver import BasicDecoder, CrossAttention
import torch
from torch import nn
import torch.nn.functional as F
import math

from attention_utils import PerceiverDecoder, LatentTransformerBlock, CrossAttentionBlock, PerceiverEncoder


class PerceiverIO(nn.Module):
    def __init__(self,
                 num_latent_heads,
                 num_cross_heads,
                 in_dim,
                 qlatent_dim,
                 size_latents,
                 num_latent_block,
                 qout_dim,
                 size_out,
                 qk_dim,
                 v_dim,
                 out_dim,
                 dim_feedforward,
                 dropout_prob
                 ) -> None:
        super(PerceiverIO, self).__init__()
        self.latent_q = nn.Parameter(torch.randn(size_latents,qlatent_dim))
        self.output_q = nn.Parameter(torch.randn(size_out, qout_dim))

        self.encoder_block = PerceiverEncoder(in_dim, qlatent_dim, size_latents=size_latents,qk_dim=qk_dim,v_dim=v_dim, num_heads=num_cross_heads, dim_feedforward=dim_feedforward, structure_output=True, dropout_prob=dropout_prob)
        self.latent_block = nn.ModuleList([LatentTransformerBlock(in_dim, qk_dim, v_dim, num_latent_heads, dim_feedforward, dropout_prob) for l in range(num_latent_block)])
        self.decoder = PerceiverDecoder(qlatent_dim, qout_dim, out_dim, qk_dim, v_dim, num_cross_heads, dim_feedforward, dropout_prob)
    
    def forward(self, x):
        #TODO: should we set nn.Parameter latent in here or in encoder ?
        # Same for output_q 
        # latent_q = self.init_latent_q() strategy for init ?
        b_size = x.shape[0]
        latent_q = self.latent_q.expand(b_size)
        output_q = self.output_q.expand(b_size)
        # Encoder block
        latent_q = self.encoder_block(x, latent_q)
        # Latent blocks
        latent_q = self.latent_block(latent_q)
        # Decoder blocks
        out = self.decoder(latent_q, output_q)

        return out