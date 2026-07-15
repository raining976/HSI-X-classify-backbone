import pytest
import torch
import torch.nn as nn

import src.models.acfnet.model as acfnet_model
from src.models.acfnet.model import (
    ACFNet,
    FusionHilbertMamba3D,
    FusionMamba2D,
    Hilbert3DScanner,
    PCAHilbert3DStem,
    RowColumnMamba2DStem,
    hilbert_scan_indices_3d,
)


class CPUMambaStub(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.input_shapes = []

    def forward(self, tokens):
        self.input_shapes.append(tuple(tokens.shape))
        return tokens


@pytest.fixture(autouse=True)
def use_cpu_mamba_stub(monkeypatch):
    monkeypatch.setattr(acfnet_model, "Mamba", CPUMambaStub)


def test_hilbert3d_indices_cover_volume_once():
    depth, height, width = 4, 4, 4
    indices = hilbert_scan_indices_3d(depth, height, width)

    assert len(indices) == depth * height * width
    assert sorted(indices) == list(range(depth * height * width))

    coordinates = [
        (index // (height * width), (index // width) % height, index % width)
        for index in indices
    ]
    distances = [
        sum(abs(left[axis] - right[axis]) for axis in range(3))
        for left, right in zip(coordinates, coordinates[1:])
    ]
    assert distances == [1] * (len(indices) - 1)


def test_hilbert3d_scanner_round_trip_is_exact():
    scanner = Hilbert3DScanner()
    feature = torch.arange(2 * 4 * 3 * 5, dtype=torch.float32).view(2, 4, 3, 5)

    sequence = scanner.scan(feature)
    restored = scanner.restore(sequence, feature.shape)

    assert sequence.shape == (2, 4 * 3 * 5, 1)
    assert torch.equal(restored, feature)


def test_hilbert_fusion_restores_feature_map_shape():
    module = FusionHilbertMamba3D(channels=8)
    feature = torch.randn(2, 8, 5, 7)

    output = module(feature)

    assert output.shape == feature.shape
    assert torch.isfinite(output).all()


def test_pca_hilbert3d_stem_scans_raw_volume_then_projects_channels():
    module = PCAHilbert3DStem(in_channels=6, hidden_dim=8)
    feature = torch.randn(2, 6, 5, 7, requires_grad=True)

    output = module(feature)
    output.mean().backward()

    assert module.mixer.input_shapes == [(2, 6 * 5 * 7, 8)]
    assert output.shape == (2, 8, 5, 7)
    assert feature.grad is not None
    assert torch.isfinite(feature.grad).all()


def test_row_column_mamba2d_stem_uses_left_to_right_and_top_to_bottom_scans():
    module = RowColumnMamba2DStem(in_channels=3, hidden_dim=8)
    feature = torch.randn(2, 3, 5, 7, requires_grad=True)
    captured = {}

    def capture_directions(_module, inputs):
        captured["directions"] = inputs[0].detach()

    handle = module.direction_fusion.register_forward_pre_hook(capture_directions)

    output = module(feature)
    handle.remove()
    output.mean().backward()

    assert not hasattr(module, "row_reverse_mixer")
    assert not hasattr(module, "column_reverse_mixer")
    assert module.row_forward_mixer.input_shapes == [(2 * 5, 7, 8)]
    assert module.column_forward_mixer.input_shapes == [(2 * 7, 5, 8)]
    assert isinstance(module.input_proj, nn.Linear)
    assert captured["directions"].shape == (2, 2 * 8, 5, 7)
    assert output.shape == (2, 8, 5, 7)
    assert feature.grad is not None
    assert torch.isfinite(feature.grad).all()


def test_acfnet_defaults_to_hilbert3d_and_can_fall_back_to_raster2d():
    hilbert_model = ACFNet(pca_channels=6, aux_channels=1, num_classes=4, hidden_dim=8)
    raster_model = ACFNet(
        pca_channels=6,
        aux_channels=1,
        num_classes=4,
        hidden_dim=8,
        fusion_scan="raster2d",
    )

    assert isinstance(hilbert_model.fusion_mamba, FusionHilbertMamba3D)
    assert isinstance(raster_model.fusion_mamba, FusionMamba2D)
    assert isinstance(hilbert_model.pca_stem, PCAHilbert3DStem)
    assert isinstance(hilbert_model.x_stem, RowColumnMamba2DStem)


@pytest.mark.parametrize("fusion_scan", ["hilbert3d", "raster2d"])
def test_acfnet_forward_shape_for_each_fusion_scan(fusion_scan):
    model = ACFNet(
        pca_channels=6,
        aux_channels=1,
        num_classes=4,
        hidden_dim=8,
        fusion_scan=fusion_scan,
    )
    pca = torch.randn(2, 6, 5, 5)
    auxiliary = torch.randn(2, 1, 5, 5)

    logits = model(pca, auxiliary)

    assert logits.shape == (2, 4)
    assert torch.isfinite(logits).all()
