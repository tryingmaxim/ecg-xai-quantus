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
)


def build_model(name: str, num_classes: int):
    name = name.lower()

    if name == "resnet18":
        m = resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    elif name == "resnet34":
        m = resnet34(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    elif name == "resnet50":
        m = resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    elif name == "resnet101":
        m = resnet101(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    elif name == "densenet121":
        m = densenet121(weights=None)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)

    elif name == "densenet169":
        m = densenet169(weights=None)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)

    elif name == "vgg16_bn":
        m = vgg16_bn(weights=None)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)

    elif name == "efficientnet_b0":
        m = efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)

    elif name == "efficientnet_b1":
        m = efficientnet_b1(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)

    elif name == "mobilenet_v2":
        m = mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)

    else:
        raise ValueError(f"Unbekanntes Modell: {name}")

    return m
