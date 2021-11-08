from argparse import ArgumentParser

import pytorch_lightning as pl
import torchmetrics
import torch
import torch.nn as nn
from monai.networks.nets.densenet import densenet121
from torch.nn import functional as F
from torch.optim import SGD, lr_scheduler, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import timm


class LitClassifier(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.save_hyperparameters()
        if self.hparams.model == "densenet121":
            self.backbone = densenet121(spatial_dims=2, in_channels=3, out_channels=2)
        elif self.hparams.model == "resnet18":
            self.backbone = timm.create_model(
                "resnet18", pretrained=True, num_classes=2
            )
        elif self.hparams.model == "resnet34":
            self.backbone = timm.create_model(
                "resnet34", pretrained=True, num_classes=2
            )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.backbone(images)
        _, pred_labels = torch.max(outputs, 1)
        train_loss = self.criterion(outputs, labels)
        train_acc = self.train_acc(pred_labels, labels)
        self.log("loss/train", train_loss, on_step=False, on_epoch=True)
        self.log("acc/train", train_acc, on_step=False, on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.backbone(images)
        _, pred_labels = torch.max(outputs, 1)
        valid_loss = self.criterion(outputs, labels)
        valid_acc = self.valid_acc(pred_labels, labels)
        self.log("loss/valid", valid_loss, on_step=False, on_epoch=True)
        self.log("acc/valid", valid_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        if self.hparams.opt == "adam":
            opt = Adam(self.parameters(), lr=self.hparams.learning_rate)
            return {
                "optimizer": opt,
                "lr_scheduler": ReduceLROnPlateau(opt),
                "monitor": "loss/valid",
            }
        elif self.hparams.opt == "sgd":
            optimizer = SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                momentum=self.hparams.momentum,
            )
            scheduler = lr_scheduler.StepLR(
                optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "loss/valid",
            }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--step_size", type=int, default=7)
        parser.add_argument("--gamma", type=float, default=0.1)
        return parser
