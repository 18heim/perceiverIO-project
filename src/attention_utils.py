import torch
from torch import nn
import torch.nn.functional as F
import math
from positional_encoding import PositionalEncoding
from icecream import ic



def scaled_dot_product(q, k, v, mask=None):
    """
        Parameters:
            mask: represents the optional masking of specific entries in the attention matrix
        Returns:
            values: output of the attention layer.
            attention: attention weights
    """
    q, k, v = (l.permute(0, 2, 1, 3) for l in (q, k , v))
    qk_dim_head = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1)) # (batch_size, heads, q_length, kv_length)
    attn_logits = attn_logits / math.sqrt(qk_dim_head)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v) # (batch_size, heads, q_length, v_dim)
    values = values.permute(0, 2, 1, 3) #(batch_size,q_length, heads,v_dim)
    return values, attention



class MultiheadAttention(nn.Module):
    """
    Basic Mutli-headed attention module.
    The scaled dot product attention allows a network to attend over a sequence.
    However, often there are multiple different aspects a sequence element wants to attend to.
    In perceiver, K,V are projected inputs, Q is not.

    Attributes
    --------
    qk_dim: int
        Q and K need to have same dimensions to compute attention weights
    v_dim: int
        V by default has the same dimension as Q and K i.e the query latent space dim but
        you could change it as to have a different output dimension.
    out_dim: int
        dimension of the output after the result of attention goes through an MLP.
    num_heads: int
        multi-head
    """
    def __init__(self,
                 in_dim,
                 qlatent_dim,
                 qk_dim,
                 v_dim,
                 out_dim,
                 num_heads,
                 ) -> None:
        super(MultiheadAttention,self).__init__()
        self._num_heads = num_heads
        self._qk_dim = qk_dim
        self._v_dim = v_dim
        self._out_dim = out_dim
        self._in_dim = in_dim
        self._qlatent_dim = qlatent_dim

        if self._v_dim % self._num_heads != 0:
            raise ValueError(f'"V Embedding dimension ({self._v_dim}) \
                must be 0 modulo number of heads ({self._num_heads}).')
        
        if self._qk_dim % self._num_heads != 0:
            raise ValueError(f'"Q/K Embedding dimension ({self._qk_dim}) \
                must be 0 modulo number of heads ({self._num_heads}).')

        self.q_proj = nn.Linear(self._qlatent_dim, self._qk_dim)
        self.k_proj = nn.Linear(self._in_dim, self._qk_dim)
 
        self.v_proj = nn.Linear(self._in_dim, self._v_dim)
        self.out_proj = nn.Linear(self._v_dim, self._out_dim)

        self._qk_dim_head = self._qk_dim // self._num_heads
        self._v_dim_head = self._v_dim // self._num_heads

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        #TODO: Refaire et l'appliquer à nos couches
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
    
    def forward(self, input_kv, input_q, mask=None):
        batch_size, kv_length, _ = input_kv.size()
        _, q_length, _ = input_q.size()
        k = self.k_proj(input_kv) #k c'est la key qui prend l'input
        v = self.v_proj(input_kv)  # value qui prend l'input
        q = self.q_proj(input_q) # q qui prend la latent array
        
        # On reshape pour avoir le nombre de heads. 
        q = q.reshape(batch_size, q_length, self._num_heads, self._qk_dim_head)  #n   : q length
        k = k.reshape(batch_size, kv_length, self._num_heads, self._qk_dim_head) #m : kv length
        v = v.reshape(batch_size, kv_length, self._num_heads, self._v_dim_head)

        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.reshape(batch_size, q_length, self._v_dim)
        o = self.out_proj(values)

        return o

class LatentTransformerBlock(nn.Module): #violet dans le schéma. Merci Mathieu pour cette indication !
    """
    Implementation of perceiverIO LatentTransformer block inspired from typical
    Transformer encoder block by Vaswani et al.
    
    attributes : 
    in_dim = qlatent_dim 
    """

    def __init__(self,
                 qlatent_dim,
                 qk_dim,
                 v_dim,
                 num_heads,
                 dim_feedforward,
                 dropout_prob) -> None:
        super(LatentTransformerBlock, self).__init__()
        
        #q_k dim c'est la projection, les deux entrées c'est in_dim et q_latent_dim. 
        self.self_attention = MultiheadAttention(in_dim=qlatent_dim,
                                            qlatent_dim=qlatent_dim,
                                            qk_dim=qk_dim,
                                            v_dim=v_dim, 
                                            out_dim=qlatent_dim, 
                                            num_heads=num_heads)
        self.norm1 = nn.LayerNorm(qlatent_dim)
        self.norm2 = nn.LayerNorm(qlatent_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear_net = nn.Sequential(
            nn.Linear(qlatent_dim, dim_feedforward),
            nn.Dropout(dropout_prob),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, qlatent_dim)
        )

    def forward(self, x, mask=None):
        # Attention part
        x_norm = self.norm1(x)
        attn_out = self.self_attention(x_norm, x_norm, mask=mask) #on fait un self attention normal. 
        x += self.dropout(attn_out)
        x = self.norm2(x)

        # MLP part
        linear_out = self.linear_net(x)
        x += self.dropout(linear_out)
        ic(x.shape)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self,
                 in_dim,
                 qlatent_dim,
                 qk_dim,
                 v_dim,
                 num_heads,
                 dim_feedforward,
                 dropout_prob) -> None:
        super(CrossAttentionBlock, self).__init__()
        self.cross_attention = MultiheadAttention(in_dim=in_dim,qlatent_dim = qlatent_dim,qk_dim=qk_dim,v_dim=v_dim,out_dim=qlatent_dim,num_heads=num_heads)

        self.normq = nn.LayerNorm(qlatent_dim)
        self.normx = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(qlatent_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear_net = nn.Sequential(
            nn.Linear(qlatent_dim, dim_feedforward),
            nn.Dropout(dropout_prob),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, qlatent_dim)
        )

    def forward(self, x, q, mask=None):
        # Attention part
        x_norm, q_norm = self.normx(x), self.normq(q)
        attn_out = self.cross_attention(x_norm, q_norm, mask=mask)
        q += self.dropout(attn_out)
        q = self.norm2(q)

        # MLP part
        linear_out = self.linear_net(q)
        q += self.dropout(linear_out)
    
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
    # TODO: Position encoding + Cross-attention
    def __init__(self,
                 in_dim,
                 qlatent_dim,
                 qk_dim,
                 v_dim,
                 num_heads,
                 dim_feedforward,
                 structure_output:bool,
                 dropout_prob :float = 0.5) -> None:
        super(PerceiverEncoder,self).__init__()
        self.structure_output = structure_output
        if self.structure_output:
            self.position_encoder = PositionalEncoding(in_dim, dropout_prob)
        self.cross_attention = CrossAttentionBlock(in_dim=in_dim,
                                                   qlatent_dim=qlatent_dim,
                                                   qk_dim=qk_dim,
                                                   v_dim=v_dim,
                                                   num_heads=num_heads,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout_prob=dropout_prob)

    def forward(self,x, q):
        ## on fait le positional encoding que si les outputs ont une structure spatiale ou séquentielle. 
        if self.structure_output :
            x = self.position_encoder(x)
        
        #cross attention:
        latents = self.cross_attention(x=x,q=q)

        return latents


class PerceiverDecoder(nn.Module):
    """
    attributes: 

    """
    def __init__(self,
                 qlatent_dim,
                 qout_dim,
                 out_dim,
                 qk_dim,
                 v_dim,
                 num_heads,
                 dim_feedforward,
                 dropout_prob
                 ) -> None:
        super(PerceiverDecoder, self).__init__()

        self.cross_attention = MultiheadAttention(in_dim=qlatent_dim,
                                                num_heads=num_heads,
                                                  qlatent_dim=qout_dim,
                                                  qk_dim=qk_dim, 
                                                  v_dim=v_dim, 
                                                  out_dim=out_dim)

        self.normq_latent = nn.LayerNorm(qlatent_dim)
        self.normq_out = nn.LayerNorm(qout_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear_net = nn.Sequential(
            nn.Linear(out_dim, dim_feedforward),
            nn.Dropout(dropout_prob),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, out_dim)
        )


    def forward(self, q, q_out, mask=None):
        # Attention part
        qlatent_norm, qout_norm = self.normq_latent(q), self.normq_out(q_out)
        attn_out = self.cross_attention(qlatent_norm, qout_norm, mask=mask)
        q_out = self.norm2(attn_out)

        # MLP part
        linear_out = self.linear_net(attn_out)
        attn_out += self.dropout(linear_out)

        return attn_out
