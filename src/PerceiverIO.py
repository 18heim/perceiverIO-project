
import torch
from icecream import ic
from torch import nn
import torch.nn.functional as F
import math
from utils import ImageInputAdapter

from attention_utils import PerceiverDecoder, PerceiverEncoder


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
                 out_dim
                 ) -> None:
        super(PerceiverIO, self).__init__()
        self.input_adapter = ImageInputAdapter(image_shape=(28,28,3), num_frequency_bands=64)
        in_dim = self.input_adapter._num_input_channels()
        self.encoder = PerceiverEncoder(in_dim, qlatent_dim, num_cross_heads, num_self_heads, dropout_prob, q_length, num_latent_blocks, structure_output)
        self.decoder = PerceiverDecoder(qlatent_dim, qout_dim,  num_cross_heads, dropout_prob, qout_length)
        self.dum_linear = nn.Linear(qout_dim, out_dim)
        
    def forward(self, x):
        # Encoder block
        x = self.input_adapter(x)
        q = self.encoder(x)
        q_out = self.decoder(q)
        q_out = q_out.mean(1)
        out = self.dum_linear(q_out)

        return out