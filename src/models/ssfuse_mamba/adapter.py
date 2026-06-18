from .model import SSFuseMamba


MODEL_NAME = "SSFuseMamba"


def build_model(config, dataset_type, device):
    hsi_channels = config.get_value('data_channels')[dataset_type]
    pca_channels = config.get_value('channels')
    aux_channels = config.get_value('lidar_or_sar_channels')[dataset_type]
    num_classes = config.get_value('out_features')[dataset_type]
    stage_depths = config.get_value('stage_depths') or [1, 2]
    net = SSFuseMamba(
        pca_channels=pca_channels,
        aux_channels=aux_channels,
        num_classes=num_classes,
        hsi_channels=hsi_channels,
        embed_dim=48,
        stem_dim=16,
        stage_depths=tuple(stage_depths),
    )
    return {"net": net}


def forward_train(bundle, batch):
    return bundle["net"](
        batch["hsi"],
        batch["hsi_pca"].squeeze(1),
        batch["aux"],
    )


def forward_eval(bundle, batch):
    return bundle["net"](
        batch["hsi"],
        batch["hsi_pca"].squeeze(1),
        batch["aux"],
    )
