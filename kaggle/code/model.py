# code/model.py
import segmentation_models_pytorch as smp
import torch.nn as nn


def build_model(
    model_name="DeepLabV3Plus", encoder="resnet101", num_classes=16, pretrained=True
):
    if model_name.lower() == "unet++":
        return smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights="imagenet" if pretrained else None,
            classes=num_classes,
            activation=None,
        )
    elif model_name.lower() == "fpn":
        return smp.FPN(
            encoder_name=encoder,
            encoder_weights="imagenet" if pretrained else None,
            classes=num_classes,
            activation=None,
        )
    else:
        return smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights="imagenet" if pretrained else None,
            classes=num_classes,
            activation=None,
        )
