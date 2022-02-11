import pytorch_lightning as pl
import torch
from torch import nn

from datasets.imagenet_dataset import ImagenetDataModule
from datasets.mnist_dataset import LitMNIST
from lightning_utils import LightningNetwork
from PerceiverIO import PerceiverIO
from utils import ImageClassificationAdapter, ImageInputAdapter, OutputAdapter

if __name__=="__main__":
    torch.cuda.empty_cache()
    #data_dir = Path('/content/tiny-imagenet-200')
    num_class = 10
    QOUT_DIM = 256
    #data = ImagenetDataModule(data_dir, image_size=64, num_workers=2, batch_size=32, pin_memory=True, setup_validation=True)
    input_adapter_params = {'type': 'ImageInputAdapter', 'image_shape':(64,64,3), 'num_frequency_bands' : 64}
    output_adapter_params = {'type' : 'ImageClassificationAdapter', 'qout_dim': QOUT_DIM, 'num_class': num_class}
    output_adapter = ImageClassificationAdapter()
    network = PerceiverIO(num_self_heads=4,
                          num_cross_heads=4,
                            in_dim=11,
                            qlatent_dim=256,
                            qout_dim=QOUT_DIM,
                            q_length=256,
                            qout_length=128,
                            num_latent_blocks=2,
                            dropout_prob=0.0,
                            structure_output=False
                          )

    params = {'type': 'Adam',
                 'lr': 0.001}
    criterion = nn.CrossEntropyLoss()
    # train_loader = data.train_dataloader()
    # val_loader = data.val_dataloader()
    model = LitMNIST('test', network, params, criterion, batch_size=128)
    #model = LightningNetwork(loss=criterion, optimizer_params=params, name='Test0', network=network)
    #logger = TensorBoardLogger(save_dir=LOG_PATH, name=model.name, version=time.asctime(), default_hp_metric=False)
    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else None,
                        #logger=logger,
                        #default_root_dir=LOG_PATH,
                        max_epochs=120)
                        #callbacks=[checkpoint_callback])
    
    #hyperparameters = conf
    #trainer.logger.log_hyperparams(hyperparameters)

    trainer.fit(model)
