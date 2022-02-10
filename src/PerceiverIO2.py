import torch
from icecream import ic
from torch import nn
import torch.nn.functional as F
import math

from attention_utils import PerceiverDecoder, LatentTransformerBlock, CrossAttentionBlock, PerceiverEncoder
from positional_encoding import PositionalEncoding




class PerceiverIO(nn.Module):
    def __init__(self,
                 num_latent_heads,
                 num_cross_heads,
                 in_dim,
                 qlatent_dim,
                 q_length,
                 num_latent_block,
                 qout_dim,
                 qout_length,
                 num_class,
                 qk_dim,
                 v_dim,
                 out_dim,
                 dim_feedforward,
                 dropout_prob,
                 structure_output
                 ) -> None:
        super(PerceiverIO, self).__init__()
        self.latent_q = nn.Parameter(torch.randn(q_length,qlatent_dim))
        self.output_q = nn.Parameter(torch.randn(qout_length, qout_dim))
        self._init_parameters()
        self.num_latent_block = num_latent_block
        self.encoder_block = PerceiverEncoder(in_dim, qlatent_dim,qk_dim=qk_dim,v_dim=v_dim, num_heads=num_cross_heads, dim_feedforward=dim_feedforward, structure_output=structure_output, dropout_prob=dropout_prob)
        self.latent_block = LatentTransformerBlock(qlatent_dim, qk_dim, v_dim, num_latent_heads, dim_feedforward, dropout_prob)
        self.decoder = PerceiverDecoder(qlatent_dim, qout_dim, out_dim, qk_dim, v_dim, num_cross_heads, dim_feedforward, dropout_prob)
        self.pos_enc = PositionalEncoding(d_model = in_dim)
        self.avpool = torch.nn.AvgPool1d(3,2)
        self.dum_linear = nn.Linear(out_dim,num_class)


    def _init_parameters(self):
        with torch.no_grad():
            self.output_q.normal_(0.0, 0.02).clamp_(-2.0, 2.0)
            self.latent_q.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, x):
        #TODO: should we set nn.Parameter latent in here or in encoder ? 
        # i don't know
        # what do you think ? 
        # Same for output_q 
        # latent_q = self.init_latent_q() strategy for init ?

        b_size = x.shape[0]
        latent_q = self.latent_q.unsqueeze(dim=0).repeat(b_size,1,1)
        output_q = self.output_q.unsqueeze(dim=0).repeat(b_size,1,1)
        # Encoder block
        latent_q = self.encoder_block(x, latent_q)
        # Latent blocks
        for _ in range(self.num_latent_block):
            latent_q = self.latent_block(latent_q)  
        # Decoder blocks
        out = self.decoder(latent_q, output_q)
        out = out.view(out.shape[0],out.shape[2],-1)
        out = self.avpool(out)
        out = out.view(out.shape[0],out.shape[2],-1)
        out = self.dum_linear(out)
     


    
        return out