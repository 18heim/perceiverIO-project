"""
attention_utils test file
"""

from icecream import ic
import torch
import sys 
sys.path.insert(1,'../src')
from attention_utils import LatentTransformerBlock, MultiheadAttention, PerceiverDecoder,PerceiverEncoder


batch_size = 32
in_dim     = 10
qlatent_dim   = 16
q_length  = 20
kv_length = 200
num_heads = 2
qk_dim = 100
v_dim= 80
olatent_dim = 12
o_length = 180

x = torch.randn(batch_size,kv_length,in_dim)
qlatent= torch.randn(batch_size,q_length,qlatent_dim)
olatent= torch.randn(batch_size,o_length,olatent_dim)


multihead = MultiheadAttention(in_dim=in_dim,
                 qlatent_dim=qlatent_dim,
                 qk_dim=qk_dim,
                 v_dim=v_dim,
                 out_dim=10,
                 num_heads=num_heads)

encoder = PerceiverEncoder(in_dim= in_dim,
                 qlatent_dim=qlatent_dim,
                 qk_dim=qk_dim,
                 v_dim=v_dim,
                 num_heads=num_heads,
                 dim_feedforward=100,
                 structure_output=True)

selfattention = LatentTransformerBlock(
                 qlatent_dim=qlatent_dim,
                 qk_dim=qk_dim,
                 v_dim=v_dim,
                 num_heads=num_heads,
                 dim_feedforward=100,
                 dropout_prob=0.5) 


decoder = PerceiverDecoder(
                 qlatent_dim=qlatent_dim,
                 qout_dim=olatent_dim,
                 out_dim=8,
                 qk_dim=qk_dim,
                 v_dim=v_dim,
                 num_heads=num_heads, 
                 dim_feedforward=2,
                 dropout_prob=0.5)
        

def test_multi_head(x, latent): 
    return multihead(x,latent)

def test_backward_multi_head(x, latent):
     with torch.autograd.set_detect_anomaly(True):
        y = multihead(x, latent)
        y = y.mean()
        y.backward()

def test_encoder(x, latent): 
    return encoder(x,latent)

def test_backward_encoder(x, latent):
    with torch.autograd.set_detect_anomaly(True):
        y = encoder(x, latent)
        y = y.mean()
        y.backward()

def test_self_attention(x):
    return selfattention(x)

def test_backward_self_attention(x):
    with torch.autograd.set_detect_anomaly(True):
        y = selfattention(x)
        y = y.mean()
        y.backward()

def test_decoder(x, latent):
    return decoder(x, latent)

def test_backward_decoder(x, latent):
    with torch.autograd.set_detect_anomaly(True):
        y = decoder(x, latent)
        y = y.mean()
        y.backward()

try : 
    o = test_multi_head(x,qlatent)
except ValueError:
    raise ValueError("ValueError multi head error")
if o.shape != (batch_size,q_length,in_dim): 
    raise ValueError("output multihead shape wrong")


o = test_encoder(x, qlatent)
if o.shape != (batch_size, q_length,qlatent_dim): 
    raise ValueError("output encoder shape wrong")

try : 
    o = test_self_attention(qlatent)
except ValueError:
    raise ValueError("ValueError latent error")


try : 
    o = test_decoder(qlatent,olatent)
except ValueError:
    raise ValueError("ValueError latent error")

test_backward_decoder(qlatent, olatent)
test_backward_encoder(x, qlatent)
test_backward_multi_head(x, qlatent)
test_backward_self_attention(qlatent)


