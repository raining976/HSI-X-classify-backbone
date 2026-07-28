import torch

from .model import FusAtNet


MODEL_NAME = "FusAtNet"


def build_model(config, dataset_type, device):
    in_channels = config.get_value('channels')
    lidar_or_sar_channels = config.get_value('lidar_or_sar_channels')[dataset_type]
    num_classes = config.get_value('out_features')[dataset_type]
    net = FusAtNet(input_channels=in_channels, input_channels2=lidar_or_sar_channels, num_classes=num_classes)
    return {"net": net}


def _to_class_logits(outputs):
    if outputs.dim() == 1:
        return outputs.unsqueeze(0)
    if outputs.dim() > 2:
        return outputs.mean(dim=tuple(range(2, outputs.dim())))
    return outputs


def forward_train(bundle, batch):
    return _to_class_logits(bundle["net"](batch["hsi_pca"].squeeze(1), batch["aux"]))


def forward_eval(bundle, batch):
    return _to_class_logits(bundle["net"](batch["hsi_pca"].squeeze(1), batch["aux"]))
