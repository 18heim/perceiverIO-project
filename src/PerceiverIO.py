
import torch
from icecream import ic
from torch import nn
import torch.nn.functional as F
import math
from utils import ImageInputAdapter

from attention_utils import PerceiverDecoder, PerceiverEncoder
from src.utils import adapter_factory

class PerceiverIO(nn.Module):
    def __init__(self,
                 num_self_heads,
                 num_cross_heads,
                 in_dim,
                 qlatent_dim,
                 qout_dim,
                 q_length,
                 qout_length,
                 num_latent_blocks,
                 dropout_prob,
                 structure_output,
                 input_adapter_params,
                 output_adapter_params,
                 ) -> None:
        super(PerceiverIO, self).__init__()
        self.input_adapter = adapter_factory(input_adapter_params)
        in_dim = self.input_adapter._num_input_channels()
        self.encoder = PerceiverEncoder(in_dim, qlatent_dim, num_cross_heads, num_self_heads, dropout_prob, q_length, num_latent_blocks, structure_output)
        self.decoder = PerceiverDecoder(qlatent_dim, qout_dim,  num_cross_heads, dropout_prob, qout_length)
        self.output_adapter = adapter_factory(output_adapter_params)
        
    def forward(self, x):
        # Encoder block
        x = self.input_adapter(x)
        q = self.encoder(x)
        q_out = self.decoder(q)
        out = self.output_adapter(q_out)

        return out