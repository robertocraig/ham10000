import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import deit_tiny_patch16_224
from torchmetrics.classification import BinaryAccuracy

class DeiTLightning(pl.LightningModule):
    def __init__(self, num_classes=2, learning_rate=1e-3):
        super(DeiTLightning, self).__init__()
        self.save_hyperparameters()

        # Carregar o DeiT pré-treinado
        self.model = deit_tiny_patch16_224(pretrained=True)

        # Ajustar a cabeça do DeiT para classificação binária
        self.model.head = nn.Linear(self.model.head.in_features, 1)  # Classificação binária (1 saída)

        # Função de perda para classificação binária
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Métrica de acurácia para tarefas binárias
        self.train_accuracy = BinaryAccuracy()
        self.val_accuracy = BinaryAccuracy()
        self.test_accuracy = BinaryAccuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['numeric_label']
        outputs = self(images).squeeze(1)  # Garante que a saída seja um vetor de previsões

        # Calcular a perda com BCEWithLogitsLoss
        loss = self.loss_fn(outputs, labels.float())

        # Converter as probabilidades para rótulos binários
        preds = torch.sigmoid(outputs) >= 0.5  # Limiar de 0.5 para classificação binária

        # Calcular a acurácia no treinamento
        acc = self.train_accuracy(preds.int(), labels.int())

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['numeric_label']
        outputs = self(images).squeeze(1)

        # Calcular a perda
        loss = self.loss_fn(outputs, labels.float())

        # Converter as probabilidades para rótulos binários
        preds = torch.sigmoid(outputs) >= 0.5

        # Calcular a acurácia na validação
        acc = self.val_accuracy(preds.int(), labels.int())

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['numeric_label']
        outputs = self(images).squeeze(1)

        # Calcular a perda
        loss = self.loss_fn(outputs, labels.float())

        # Converter as probabilidades para rótulos binários
        preds = torch.sigmoid(outputs) >= 0.5

        # Calcular a acurácia no teste
        acc = self.test_accuracy(preds.int(), labels.int())

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
