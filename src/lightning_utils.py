import time
from pathlib import Path
from typing import Optional, Dict, Any
import importlib

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.distributions import Categorical
from torch.functional import norm
from torch.utils.data import DataLoader, TensorDataset, random_split
from icecream import ic
import torchmetrics as tm

def optim_factory(model: nn.Module, optimizer_params: Dict[str, Any]):
    """Construct Optimizer object from input serialized parameters dict."""
    optimizer_name = optimizer_params['type']
    module = importlib.import_module('torch.optim')
    optimizer_class = getattr(module, optimizer_name)
    return optimizer_class(params=model.parameters(),
                           **{k: v for k, v in optimizer_params.items() if k != 'type'})


class LightningNetwork(pl.LightningModule):
    def __init__(self,
                 name,
                 network: nn.Module,
                 optimizer_params,
                 loss):
        super().__init__()
        self.model = network
        self.loss = loss
        self.name = name
        self.optimizer_params = optimizer_params
        self.acc = tm.classification.accuracy.Accuracy()

    def forward(self,x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        """ Définit l'optimiseur """
        optimizer = optim_factory(self.model, self.optimizer_params)
        return optimizer

    def training_step(self,batch,batch_idx):
        x, y = batch
        batch_size, channels, height, width = x.size()
        x = x.permute(0,2,3,1)
        x = x.view(batch_size, -1, channels)
        yhat = self(x).view(batch_size,-1)
        loss = self.loss(yhat,y)
        acc = self.acc(yhat.argmax(-1), y)
        logs = {"loss":loss,"accuracy":acc,"nb":len(x)}
        self.log("accuracy",acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("training_loss",loss, on_step=False,on_epoch=True)
        return logs

    def validation_step(self,batch,batch_idx):
        """ une étape de validation
        doit retourner un dictionnaire"""
        x, y = batch
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, -1, channels)
        yhat = self(x).view(batch_size,-1)
        loss = self.loss(yhat,y)
        acc = self.acc(yhat.argmax(-1), y)
        logs = {"loss":loss,"accuracy":acc,"nb":len(x)}
        self.log("val_accuracy", acc/len(x), on_step=False, on_epoch=True, prog_bar=True)
        self.log("validation_loss",loss, on_step=False,on_epoch=True)
        return logs

    def test_step(self,batch,batch_idx):
        """ une étape de test """
        x, y = batch
        yhat = self(x)
        loss = self.loss(yhat,y)
        acc = (yhat.argmax(1)==y).sum()
        logs = {"loss":loss,"accuracy":acc,"nb":len(x)}
        return logs

    def test_epoch_end(self, outputs):
        pass
        