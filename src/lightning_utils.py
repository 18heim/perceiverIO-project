from typing import Any, Dict, Optional, Tuple
import pytorch_lightning as pl
import torch
import torchmetrics as tm
from torch import nn
import importlib
from torchmetrics import Accuracy
from icecream import ic


def optim_factory(model: nn.Module, optimizer_params: Dict[str, Any]):
    """Construct Optimizer object from input serialized parameters dict."""
    optimizer_name = optimizer_params['type']
    module = importlib.import_module('torch.optim')
    optimizer_class = getattr(module, optimizer_name)
    return optimizer_class(params=model.parameters(),
                           **{k: v for k, v in optimizer_params.items() if k != 'type'})


class LightningClassificationNetwork(pl.LightningModule):
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
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def forward(self,x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        """ Définit l'optimiseur """
        optimizer = optim_factory(self.model, self.optimizer_params)
        return optimizer

    def training_step(self,batch,batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        preds = logits.argmax(dim=-1)
        acc = self.train_accuracy(preds, y)
        self.log("train_accuracy", acc, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        logs = {"loss": loss, "accuracy": acc}
        return logs

    def validation_step(self,batch,batch_idx):
        """ une étape de validation
        doit retourner un dictionnaire"""
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        preds = logits.argmax(dim=-1)
        acc = self.val_accuracy(preds, y)
        self.log("val_accuracy", acc, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        logs = {"loss": loss, "accuracy": acc}
        return logs

    def test_step(self,batch,batch_idx):
        """ une étape de test """
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        preds = logits.argmax(dim=-1)
        # acc = (yhat.argmax(1)==y).sum()
        acc = self.test_accuracy(preds, y)
        logs = {"loss":loss,"accuracy":acc,"nb":len(x)}
        return logs

    def training_epoch_end(self, outputs):
        total_loss = sum([o['loss'] for o in outputs]) / len(outputs)
        total_accuracy = sum([o['accuracy'] for o in outputs]) / len(outputs)
        self.logger.experiment.add_scalar("Train loss", total_loss.item(), self.current_epoch)
        self.logger.experiment.add_scalar("Train accuracy", total_accuracy.item(), self.current_epoch)

    def validation_epoch_end(self, outputs):
        total_loss = sum([o['loss'] for o in outputs]) / len(outputs)
        total_accuracy = sum([o['accuracy'] for o in outputs]) / len(outputs)
        self.logger.experiment.add_scalar("Val loss", total_loss.item(), self.current_epoch)
        self.logger.experiment.add_scalar("Val accuracy", total_accuracy.item(), self.current_epoch)

    def test_epoch_end(self, outputs):
        total_loss = sum([o['loss'] for o in outputs]) / len(outputs)
        total_accuracy = sum([o['accuracy'] for o in outputs]) / len(outputs)
        self.logger.experiment.add_scalar("Test loss", total_loss.item(), self.current_epoch)
        self.logger.experiment.add_scalar("Test accuracy", total_accuracy.item(), self.current_epoch)
        

class LightningRegressionNetwork(pl.LightningModule):
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
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def forward(self,x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        """ Définit l'optimiseur """
        optimizer = optim_factory(self.model, self.optimizer_params)
        return optimizer

    def training_step(self,batch,batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        preds = logits.argmax(dim=-1)
        acc = self.train_accuracy(preds, y)
        self.log("train_accuracy", acc, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        logs = {"loss": loss, "accuracy": acc}
        return logs

    def validation_step(self,batch,batch_idx):
        """ une étape de validation
        doit retourner un dictionnaire"""
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        preds = logits.argmax(dim=-1)
        acc = self.val_accuracy(preds, y)
        self.log("val_accuracy", acc, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        logs = {"loss": loss, "accuracy": acc}
        return logs

    def test_step(self,batch,batch_idx):
        """ une étape de test """
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        preds = logits.argmax(dim=-1)
        # acc = (yhat.argmax(1)==y).sum()
        acc = self.test_accuracy(preds, y)
        logs = {"loss":loss,"accuracy":acc,"nb":len(x)}
        return logs

    def training_epoch_end(self, outputs):
        total_loss = sum([o['loss'] for o in outputs]) / len(outputs)
        total_accuracy = sum([o['accuracy'] for o in outputs]) / len(outputs)
        self.logger.experiment.add_scalar("Train loss", total_loss.item(), self.current_epoch)
        self.logger.experiment.add_scalar("Train accuracy", total_accuracy.item(), self.current_epoch)

    def validation_epoch_end(self, outputs):
        total_loss = sum([o['loss'] for o in outputs]) / len(outputs)
        total_accuracy = sum([o['accuracy'] for o in outputs]) / len(outputs)
        self.logger.experiment.add_scalar("Val loss", total_loss.item(), self.current_epoch)
        self.logger.experiment.add_scalar("Val accuracy", total_accuracy.item(), self.current_epoch)

    def test_epoch_end(self, outputs):
        total_loss = sum([o['loss'] for o in outputs]) / len(outputs)
        total_accuracy = sum([o['accuracy'] for o in outputs]) / len(outputs)
        self.logger.experiment.add_scalar("Test loss", total_loss.item(), self.current_epoch)
        self.logger.experiment.add_scalar("Test accuracy", total_accuracy.item(), self.current_epoch)