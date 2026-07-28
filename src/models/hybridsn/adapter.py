from .model import HybridSN


MODEL_NAME = "HybridSN"


def build_model(config, dataset_type, device):
    channels = config.get_value('channels')
    window_size = config.get_value('windowSize')
    if channels != 180 or window_size != 11:
        raise ValueError(
            "HybridSN 当前实现硬编码为 channels=180 且 window_size=11；"
            f"当前配置为 channels={channels}, window_size={window_size}。"
            "请单独用匹配配置运行 HybridSN，或从 model_name 列表中移除 HybridSN。"
        )
    num_classes = config.get_value('out_features')[dataset_type]
    net = HybridSN(num_classes)
    return {"net": net}


def forward_train(bundle, batch):
    return bundle["net"](batch["hsi_pca"])


def forward_eval(bundle, batch):
    return bundle["net"](batch["hsi_pca"])
