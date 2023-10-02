import torch.nn as nn
import torchvision.models as models


class CNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet50 = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        for param in resnet50.parameters():
            param.requires_grad_(False)

        layers = list(resnet50.children())[:-1]
        self.resnet = nn.Sequential(*layers)
        self.embed = nn.Linear(resnet50.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
