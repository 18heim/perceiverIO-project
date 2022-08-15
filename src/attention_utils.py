import torch
from torch import nn
import torch.nn.functional as F
import math
from einops import repeat
from icecream import ic

class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self:
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


def mlp(num_channels: int):
    return nn.Sequential(
            nn.LayerNorm(num_channels),
            nn.Linear(num_channels, num_channels),
            nn.GELU(),
            nn.Linear(num_channels, num_channels)
        )


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 in_dim,
                 qlatent_dim,
                 num_heads,
                 dropout_prob
                 ) -> None:
        super(MultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=qlatent_dim, kdim=in_dim, vdim=in_dim,
                                               num_heads=num_heads, dropout=dropout_prob,
                                               batch_first=True)

    def forward(self, input_kv, input_q, mask = None):
        return self.attention(input_q, input_kv, input_kv, attn_mask=mask)[0]


class SelfAttention(nn.Module):
    def __init__(self,
                 qlatent_dim,
                 num_heads,
                 dropout_prob
                 ) -> None:
        super(SelfAttention, self).__init__()
        self.norm = nn.LayerNorm(qlatent_dim)
        self.attention = nn.MultiheadAttention(embed_dim=qlatent_dim, kdim=qlatent_dim, vdim=qlatent_dim,
                                               num_heads=num_heads, dropout=dropout_prob,
                                               batch_first=True)

    def forward(self, x, mask = None):
        x = self.norm(x)
        return self.attention(x, x, x, attn_mask=mask)[0]


class SelfAttentionBlock(nn.Module): #violet dans le schéma. Merci Mathieu pour cette indication !
    """
    Implementation of perceiverIO LatentTransformer block inspired from typical
    Transformer encoder block by Vaswani et al.
    attributes : 
    in_dim = qlatent_dim 
    """
    def __init__(self,
                 qlatent_dim,
                 num_heads,
                 dropout_prob,
                 num_layers) -> None:
        super(SelfAttentionBlock, self).__init__()
        
        #q_k dim c'est la projection, les deux entrées c'est in_dim et q_latent_dim. 
        layers = [SelfAttention(qlatent_dim, num_heads, dropout_prob) for _ in range(num_layers)]
        self.self_attention = mySequential(*layers)
        self.norm1 = nn.LayerNorm(qlatent_dim)
        self.norm2 = nn.LayerNorm(qlatent_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear_net = mlp(num_channels=qlatent_dim)

    def forward(self, q, mask=None):
        # Attention part
        q = self.norm1(q)
        attn_out = self.self_attention(q , mask)
        q = q + attn_out
        # MLP part
        linear_out = self.linear_net(q)
        q = q + self.dropout(linear_out)
        return q


class CrossAttentionBlock(nn.Module):
    def __init__(self,
                 in_dim,
                 qlatent_dim,
                 num_heads,
                 dropout_prob) -> None:
        super(CrossAttentionBlock, self).__init__()
        self.cross_attention = MultiHeadAttention(in_dim=in_dim,
                                                  qlatent_dim = qlatent_dim,
                                                  num_heads=num_heads,
                                                  dropout_prob=dropout_prob)
        self.normq = nn.LayerNorm(qlatent_dim)
        self.normx = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear_net = mlp(qlatent_dim)

    def forward(self, x, q, mask=None):
        # Attention part
        x_norm, q_norm = self.normx(x), self.normq(q)
        attn_out = self.cross_attention(x_norm, q_norm, mask=mask)
        q = q + attn_out
        # MLP part
        linear_out = self.linear_net(q)
        q = q + self.dropout(linear_out)
        return q

class LatentTransformerBlock(nn.Module): #violet dans le schéma. Merci Mathieu pour cette indication !
    """
    """
    def __init__(self,
                 in_dim,
                 qlatent_dim,
                 num_cross_heads,
                 num_self_heads,
                 num_self_layers,
                 dropout_prob) -> None:
        super(LatentTransformerBlock, self).__init__()
        
        #q_k dim c'est la projection, les deux entrées c'est in_dim et q_latent_dim. 
        self.self_attention = SelfAttentionBlock(qlatent_dim,
                                                 num_self_heads,
                                                 dropout_prob,
                                                 num_self_layers)
        self.cross_attention = CrossAttentionBlock(in_dim,
                                                   qlatent_dim,
                                                   num_cross_heads,
                                                   dropout_prob)

    def forward(self, x, q, mask=None):
        q = self.cross_attention(x, q ,mask)
        q = self.self_attention(q, mask)
        return q

class PerceiverEncoder(nn.Module):
    """Perceiver encoder module. Consists of two components: cross-attention
    module that maps an input tensor and a trainable latent tensor to a latent
    tensor and a stacked Transformer blocks with shared weights.
    Attributes
    ----------
    structure_output : bool
        if true then we do the positional encoding
    num_blocks : int
        le nombre de block pour le processing,
    times_per_block: int
        dans chaque bloque de processing cela définit lenombre de multiheadattention qu'on va faire
    """
    # Position encoding + Cross-attention
    def __init__(self,
                 in_dim,
                 qlatent_dim,
                 num_cross_heads,
                 num_self_heads,
                 dropout_prob,
                 q_length,
                 num_latent_blocks: int = 0,
                 tie_weights: bool = True,
                 num_self_layers = 1) -> None:
        super(PerceiverEncoder,self).__init__()
        self.tie_weights = tie_weights
        self.latent_q = nn.Parameter(torch.randn(q_length,qlatent_dim))
        self._init_parameters()
        self.num_latent_blocks = num_latent_blocks
        self.encode = LatentTransformerBlock(in_dim,
                                             qlatent_dim,
                                             num_cross_heads,
                                             num_self_heads,
                                             num_self_layers,
                                             dropout_prob)

        if self.num_latent_blocks > 0:
            if self.tie_weights:
                self.latent_block = LatentTransformerBlock(in_dim,
                                                       qlatent_dim,
                                                       num_cross_heads,
                                                       num_self_heads,
                                                       num_self_layers,
                                                       dropout_prob)
            else:
                self.latent_block = nn.ModuleList()
                for i in range(self.num_latent_blocks):
                    self.latent_block.append(LatentTransformerBlock(in_dim,
                                                qlatent_dim,
                                                num_cross_heads,
                                                num_self_heads,
                                                num_self_layers,
                                                dropout_prob) )
    def _init_parameters(self):
        with torch.no_grad():
            self.latent_q.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, x, mask=None):
        b = x.shape[0]
        latent_q = repeat(self.latent_q, "... -> b ...", b=b)
        q = self.encode(x, latent_q, mask)
        if self.num_latent_blocks > 0:
            if self.tie_weights:
                for _ in range(self.num_latent_blocks - 1):
                    q = self.latent_block(x, q)
            else:
                for layer in self.latent_block:
                    q = layer(x,q)
        return q


class PerceiverDecoder(nn.Module):
    """
    attributes: 

    """
    def __init__(self,
                 qlatent_dim,
                 qout_dim,
                 num_cross_heads,
                 dropout_prob,
                 qout_length
                 ) -> None:
        super(PerceiverDecoder, self).__init__()
        self.output_q = nn.Parameter(torch.empty(qout_length, qout_dim))
        self._init_parameters()
        self.cross_attention = CrossAttentionBlock(in_dim=qlatent_dim,
                                                   qlatent_dim=qout_dim,
                                                   num_heads=num_cross_heads,
                                                   dropout_prob=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear_net = mlp(num_channels=qout_dim)

    def _init_parameters(self):
        with torch.no_grad():
            self.output_q.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, q, mask=None):
        b = q.shape[0]
        q_out = repeat(self.output_q, "... -> b ...", b=b)
        q_out = self.cross_attention(q, q_out, mask=mask)
        # MLP part
        linear_out = self.linear_net(q_out)
        q_out = q_out + self.dropout(linear_out)

        return q_out
