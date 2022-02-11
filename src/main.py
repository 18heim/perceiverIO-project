from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch import nn

from datasets.imagenet_dataset import ImagenetDataModule
from datasets.mnist_dataset import MNISTDataModule
from lightning_utils import LightningClassificationNetwork
from PerceiverIO import PerceiverIO
from utils import ImageClassificationAdapter, ImageInputAdapter, OutputAdapter

if __name__=="__main__":
    #conf = OmegaConf.load('config/config_mnist.yaml')
    #data = MNISTDataModule(batch_size=conf.batch_size)

    conf = OmegaConf.load('config/config_imagenet.yaml')
    data = ImagenetDataModule(Path(conf.data_dir), image_size=conf.image_shape, num_workers=conf.num_workers, batch_size=conf.batch_size, pin_memory=True, setup_validation=True)
    network = PerceiverIO(num_self_heads=conf.num_self_heads,
                          num_cross_heads=conf.num_cross_heads,
                            in_dim=11,
                            qlatent_dim=conf.qlatent_dim,
                            qout_dim=conf.qout_dim,
                            q_length=conf.q_length,
                            qout_length=conf.qout_length,
                            num_latent_blocks=conf.num_latent_blocks,
                            dropout_prob=conf.dropout_prob,
                            input_adapter_params=conf.input_adapter_params,
                            output_adapter_params=conf.output_adapter_params,
                          )

    criterion = nn.CrossEntropyLoss()
    # train_loader = data.train_dataloader()
    # val_loader = data.val_dataloader()
    model = LightningClassificationNetwork(loss=criterion, optimizer_params=conf.optim_params, name=conf.name, network=network)
    #logger = TensorBoardLogger(save_dir=LOG_PATH, name=model.name, version=time.asctime(), default_hp_metric=False)
    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else None,
                        #logger=logger,
                        #default_root_dir=LOG_PATH,
                        max_epochs=conf.max_epochs)
                        #callbacks=[checkpoint_callback])
    
    #hyperparameters = conf
    #trainer.logger.log_hyperparams(hyperparameters)

    trainer.fit(model, data)
