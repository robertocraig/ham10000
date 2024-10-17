import torchvision.models as models
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy

class ResNetLightning(pl.LightningModule):
    def __init__(self, num_classes=2, learning_rate=1e-3):
        super(ResNetLightning, self).__init__()
        self.save_hyperparameters()

        # Carregar o modelo ResNet18 pré-treinado
        self.model = models.resnet18(pretrained=True)
        
        # Congelar as camadas até layer3
        for name, param in self.model.named_parameters():
            # if "layer4" not in name:  # Congela até layer3
            if "layer3" not in name and "layer4" not in name:  # Congela até layer2
                param.requires_grad = False

        # Alterar para uso em classificação binária
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)  # Saída única para classificação binária

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
        loss = self.loss_fn(outputs, labels.float())  # BCEWithLogitsLoss espera rótulos float

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
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',  # O scheduler ajusta a taxa de aprendizado com base na validação
                'interval': 'epoch',
                'frequency': 1
            }
        }
