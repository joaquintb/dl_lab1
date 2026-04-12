import torch.nn as nn
from torchvision.models import resnet18

from vgg_from_scratch import vgg16


def build_model(args, num_classes):
    model_name = getattr(args, "model", "vgg16").lower()

    if model_name == "vgg16":
        model = vgg16(num_classes=num_classes)

        if getattr(args, "dropout", None) is not None:
            for m in model.classifier.modules():
                if isinstance(m, nn.Dropout):
                    m.p = float(args.dropout)

        return model

    if model_name == "resnet18":
        # Trained from scratch: no pretrained weights
        model = resnet18(weights=None)

        # Classification head
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

        return model

    raise ValueError(
        f"Unsupported model '{model_name}'. "
        f"Supported: vgg16, resnet18"
    )