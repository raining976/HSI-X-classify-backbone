import torch
import torch.nn as nn

import src.trainer as trainer


class FakeMamba(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_inner = 16
        self.d_state = 4
        self.d_conv = 3
        self.dt_rank = 1

    def forward(self, tokens):
        return tokens


class ReusedMambaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mixer = FakeMamba()

    def forward(self, pca, _aux):
        tokens = pca.flatten(2).transpose(1, 2)
        return self.mixer(self.mixer(tokens)).mean(dim=(1, 2), keepdim=True)


class ComputeAdapter:
    def forward_train(self, model_bundle, batch):
        return model_bundle["net"](batch["hsi_pca"], batch["aux"])


def test_compute_estimator_counts_each_call_to_a_shared_mamba(monkeypatch):
    net = ReusedMambaNet()
    loader = [
        (
            torch.randn(1, 8, 2, 3),
            torch.randn(1, 8, 2, 3),
            torch.randn(1, 1, 2, 3),
            torch.zeros(1, dtype=torch.long),
        )
    ]
    monkeypatch.setattr(
        trainer,
        "_is_mamba_module",
        lambda module: isinstance(module, FakeMamba),
    )
    single_call_macs = trainer._estimate_mamba_macs(
        net.mixer,
        (torch.randn(1, 6, 8),),
    )

    summary = trainer._estimate_model_compute(
        ComputeAdapter(),
        {"net": net},
        net,
        loader,
        torch.device("cpu"),
    )

    assert single_call_macs == 5_184
    assert "MACs=10.368K" in summary
    assert "Mamba=10.368K" in summary
