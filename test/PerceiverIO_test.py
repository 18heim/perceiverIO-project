
import torch
import sys 
sys.path.insert(1,'../src')
from PerceiverIO import PerceiverIO


batch_size = 32
in_dim     = 3
qlatent_dim   = 16
q_length  = 20
kv_length = 200
num_heads = 2
qk_dim = 100
v_dim= 80
qout_dim = 12
qout_length = 180

x = torch.randn(batch_size,kv_length,in_dim)



perceiver = PerceiverIO(num_latent_heads=4,
                 num_cross_heads=2,
                 in_dim=in_dim,
                 qlatent_dim=qlatent_dim,
                 q_length = q_length,
                 num_latent_block=3,
                 qout_dim = qout_dim,
                 qout_length= qout_length,
                 qk_dim=qk_dim,
                 v_dim=v_dim,
                 out_dim=8,
                 dim_feedforward=2,
                 dropout_prob=0.5,
                 structure_output=False)
                 
def test_perceiver(x):
    return perceiver(x)


try : 
    o = test_perceiver(x)
except ValueError:
    raise ValueError("ValueError latent error")

              