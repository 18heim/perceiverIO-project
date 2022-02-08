"""
attention_utils test file
"""

"""
attention_utils test file
"""


import torch
import sys 
sys.path.insert(1,'../src')
from attention_utils import MultiheadAttention, PerceiverDecoder,PerceiverEncoder

batch_size = 32
in_dim     = 10
qlatent_dim   = 16
q_length  = 20
kv_length = 200
num_heads = 2
qk_dim = 100
v_dim= 80

x = torch.randn(batch_size,kv_length,in_dim)
qlatent= torch.randn(batch_size,q_length, qlatent_dim)

#in dim = C
# q latent dim  = D
#
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
                 structure_output=False)

def test_multi_head(x,latent): 
    return multihead(x,latent)

def test_encoder(x,latent): 
    return encoder(x,latent)


# try : 
#     o = test_multi_head(x,qlatent)
# except ValueError:
#     raise ValueError("ValueError multi head error")
# if o.shape != [batch_size,q_length,in_dim]: 
#     raise ValueError("output multihead shape wrong")

try : 
    o = test_encoder(x,qlatent)
except ValueError:
    raise ValueError("ValueError encoder error")
  
  

# if o.shape != batch_size,q_length, heads,v_dim


# Comme output 32, 180, 8