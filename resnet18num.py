import torchvision.models as models
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy

class ResNetWithFeaturesLightning(pl.LightningModule):
    def __init__(self, num_features, num_classes=2, learning_rate=1e-3):
        super(ResNetWithFeaturesLightning, self).__init__()
        self.save_hyperparameters()

        # Carregar o modelo ResNet18 pré-treinado
        self.model = models.resnet18(pretrained=True)
        
        # Congelar as camadas até layer3
        for name, param in self.model.named_parameters():
            if "layer4" not in name:  # Congela até layer3
                param.requires_grad = False

        # Extrair o número de recursos da camada FC da ResNet
        num_ftrs = self.model.fc.in_features

        # Remover a camada FC da ResNet
        self.model.fc = nn.Identity()

        # Camada para processar as características numéricas
        self.feature_layers = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Combinar recursos da imagem e características numéricas
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs + 128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)  # Saída única para classificação binária
        )

        # Função de perda para classificação binária
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Métrica de acurácia para tarefas binárias
        self.train_accuracy = BinaryAccuracy()
        self.val_accuracy = BinaryAccuracy()
        self.test_accuracy = BinaryAccuracy()

    def forward(self, image, numerical_features):
        # Passar a imagem pela ResNet
        img_features = self.model(image)

        # Passar as características numéricas pelas camadas definidas
        num_features = self.feature_layers(numerical_features)

        # Concatenar as características
        combined_features = torch.cat((img_features, num_features), dim=1)

        # Passar pelas camadas finais para classificação
        output = self.classifier(combined_features)

        return output

    # Os métodos training_step, validation_step e test_step devem ser atualizados para lidar com as duas entradas

    def training_step(self, batch, batch_idx):
        images = batch['image']
        numerical_features = batch['numerical_features']
        labels = batch['numeric_label']

        outputs = self(images, numerical_features).squeeze(1)

        loss = self.loss_fn(outputs, labels.float())

        preds = torch.sigmoid(outputs) >= 0.5

        acc = self.train_accuracy(preds.int(), labels.int())

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images = batch['image']
        numerical_features = batch['numerical_features']
        labels = batch['numeric_label']

        outputs = self(images, numerical_features).squeeze(1)

        loss = self.loss_fn(outputs, labels.float())

        preds = torch.sigmoid(outputs) >= 0.5

        acc = self.val_accuracy(preds.int(), labels.int())

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        images = batch['image']
        numerical_features = batch['numerical_features']
        labels = batch['numeric_label']

        outputs = self(images, numerical_features).squeeze(1)

        loss = self.loss_fn(outputs, labels.float())

        preds = torch.sigmoid(outputs) >= 0.5

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
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
