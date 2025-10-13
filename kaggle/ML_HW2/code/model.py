import segmentation_models_pytorch as smp


def build_model(
    model_name="Unet++", encoder="efficientnet-b5", num_classes=16, pretrained=True
):
    enc_w = "imagenet" if pretrained else None
    name = model_name.lower()
    if name in ["unet++", "unetplusplus", "unetpp", "unet_plus_plus"]:
        return smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights=enc_w,
            classes=num_classes,
            activation=None,
        )
    if name in ["deeplabv3+", "deeplabv3plus", "deeplabv3_plus"]:
        return smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=enc_w,
            classes=num_classes,
            activation=None,
        )
    if name in ["fpn"]:
        return smp.FPN(
            encoder_name=encoder,
            encoder_weights=enc_w,
            classes=num_classes,
            activation=None,
        )
    raise ValueError(f"Unknown model_name: {model_name}")
