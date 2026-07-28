from .model import ACFNet


MODEL_NAME = "ACFNet"


def build_model(config, dataset_type, device):
    pca_channels = config.get_value('channels')
    aux_channels = config.get_value('lidar_or_sar_channels')[dataset_type]
    num_classes = config.get_value('out_features')[dataset_type]
    attention_mode = config.get_value('acfnet_attention_mode') or 'mutual_consistency'
    num_interaction_layers = config.get_value('acfnet_num_interaction_layers') or 2
    fusion_scan = config.get_value('acfnet_fusion_scan') or 'hilbert3d'
    d_state = config.get_value('acfnet_d_state')
    if d_state is None:
        d_state = 16
    use_concentration = config.get_value('acfnet_use_concentration')
    if use_concentration is None:
        use_concentration = True
    net = ACFNet(
        pca_channels=pca_channels,
        aux_channels=aux_channels,
        num_classes=num_classes,
        hidden_dim=64,
        fusion_scan=fusion_scan,
        attention_mode=attention_mode,
        num_interaction_layers=num_interaction_layers,
        d_state=d_state,
        use_concentration=use_concentration,
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
