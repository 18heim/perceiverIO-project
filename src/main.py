from pathlib import Path
import torch
from torch import nn

from datasets.imagenet_dataset import ImagenetDataModule
from lightning_utils import LightningNetwork
from PerceiverIO import PerceiverIO
import pytorch_lightning as pl

if __name__=="__main__":
    data_dir = Path('/content/tiny-imagenet-200')
    num_class = 100000
    data = ImagenetDataModule(data_dir, image_size=64, num_workers=2, batch_size=32, pin_memory=True, setup_validation=True)
    network = PerceiverIO(num_cross_heads=4,
                          num_latent_heads=4,
                          in_dim=3,
                          qlatent_dim=128,
                          q_length = 128,
                          num_latent_block=32,
                          qout_dim=32,
                          qout_length=1,
                          v_dim = 128,
                          qk_dim = 128,
                          out_dim = num_class,
                          dim_feedforward=256,
                          dropout_prob=0.0,
                          structure_output=True
                          )

    params = {'type': 'Adam',
                 'lr': 0.001}
    criterion = nn.CrossEntropyLoss()
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()

    model = LightningNetwork(loss=criterion, optimizer_params=params, name='Test0', network=network)
    #logger = TensorBoardLogger(save_dir=LOG_PATH, name=model.name, version=time.asctime(), default_hp_metric=False)
    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else None,
                        #logger=logger,
                        #default_root_dir=LOG_PATH,
                        max_epochs=120)
                        #callbacks=[checkpoint_callback])
    
    #hyperparameters = conf
    #trainer.logger.log_hyperparams(hyperparameters)

    trainer.fit(model, train_loader, val_dataloaders=val_loader)