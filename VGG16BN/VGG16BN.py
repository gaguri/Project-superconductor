import torch
import torchvision.models as models
import torch.nn as nn

def modified_model():
    # Загрузка предобученной модели VGG16 с Batch Normalization
    vgg16_bn = models.vgg16_bn(weights='VGG16_BN_Weights.DEFAULT')

    # Заморозка предобученных слоев
    for param in vgg16_bn.parameters():
        param.requires_grad = False

    # Изменение последнего слоя
    num_features = vgg16_bn.classifier[6].in_features  # Число входных признаков в последнем слое
    vgg16_bn.classifier[6] = nn.Linear(num_features, 4)  # Заменяем последний слой на новый

    return vgg16_bn
