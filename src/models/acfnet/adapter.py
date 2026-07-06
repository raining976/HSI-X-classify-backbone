from .model import ACFNet


MODEL_NAME = "ACFNet"


def build_model(config, dataset_type, device):
    pca_channels = config.get_value('channels')
    aux_channels = config.get_value('lidar_or_sar_channels')[dataset_type]
    num_classes = config.get_value('out_features')[dataset_type]
    net = ACFNet(
        pca_channels=pca_channels,
        aux_channels=aux_channels,
        num_classes=num_classes,
        hidden_dim=64,
    )
    return {"net": net}


def forward_train(bundle, batch):
    return bundle["net"](
        batch["hsi_pca"].squeeze(1),
        batch["aux"],
    )


def forward_eval(bundle, batch):
    return bundle["net"](
        batch["hsi_pca"].squeeze(1),
        batch["aux"],
    )
