import torch.nn as nn
from torchvision.models import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    densenet121,
    densenet169,
    vgg16_bn,
    efficientnet_b0,
    efficientnet_b1,
    mobilenet_v2,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    DenseNet121_Weights,
    DenseNet169_Weights,
    VGG16_BN_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    MobileNet_V2_Weights,
)


def build_model(name: str, num_classes: int, pretrained: bool = True):
    name = name.lower()

    if name == "resnet18":
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    elif name == "resnet34":
        m = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    elif name == "resnet50":
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    elif name == "resnet101":
        m = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    elif name == "densenet121":
        m = densenet121(
            weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        )
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)

    elif name == "densenet169":
        m = densenet169(
            weights=DenseNet169_Weights.IMAGENET1K_V1 if pretrained else None
        )
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)

    elif name == "vgg16_bn":
        m = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)

    elif name == "efficientnet_b0":
        m = efficientnet_b0(
            weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)

    elif name == "efficientnet_b1":
        m = efficientnet_b1(
            weights=EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None
        )
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)

    elif name == "mobilenet_v2":
        m = mobilenet_v2(
            weights=MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        )
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)

    else:
        raise ValueError(f"Unbekanntes Modell: {name}")

    return m
