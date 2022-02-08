from pathlib import Path
import torch
from torch import nn

from datasets.imagenet_dataset import ImagenetDataModule
from lightning_utils import LightningNetwork
from PerceiverIO import PerceiverIO

if __name__=="__main__":
    data_dir = Path('./tiny-imagenet-200')
    num_class = 200
    data = ImagenetDataModule(data_dir, 64, 2, 32, True, True)
    network = PerceiverIO(num_cross_heads=2,
                          num_latent_heads=4,
                          in_dim=3,
                          qlatent_dim=64,
                          q_length = 128,
                          num_latent_block=10,
                          qout_dim=num_class,
                          qout_length=1,
                          qk_dim=32,
                          dim_feedforward=32,
                          dropout_prob=0.2)

    params = {'type': 'Adam',
                 'lr': 0.0001}
    criterion = nn.CrossEntropyLoss()
    lightning_module = LightningNetwork(loss=criterion, optimizer_params=params, name='Test0', model=network)
