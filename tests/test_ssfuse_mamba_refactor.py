import torch
import torch.nn as nn
import pytest

import src.models.ssfuse_mamba.model as ssfuse_model
from src.models.ssfuse_mamba.model import (
    AuxSpatialMamba2DBlock,
    AuxiliarySpatialStem,
    CovCrossAttention2d,
    PCAHilbertVolumeBlock,
    PCAXStage,
    PCASpatialStem,
    SSFuseMamba,
    _center_spiral_scan_indices_2d,
)


class CPUMambaStub(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, tokens):
        return tokens


@pytest.fixture(autouse=True)
def use_cpu_mamba_stub(monkeypatch):
    monkeypatch.setattr(ssfuse_model, "Mamba", CPUMambaStub)


def test_center_spiral_indices_cover_grid_once():
    indices = _center_spiral_scan_indices_2d(height=5, width=7)

    assert len(indices) == 5 * 7
    assert sorted(indices) == list(range(5 * 7))
    assert indices[0] == (5 // 2) * 7 + (7 // 2)


def test_spatial_stems_use_lightweight_residual_blocks():
    pca_stem = PCASpatialStem(in_channels=30, embed_dim=16)
    aux_stem = AuxiliarySpatialStem(in_channels=1, embed_dim=16)

    assert hasattr(pca_stem, "residual")
    assert hasattr(aux_stem, "residual")


def test_cov_cross_attention_2d_enhances_context_to_query_shape():
    module = CovCrossAttention2d(query_channels=16, context_channels=8, attn_channels=48)
    module.eval()
    query = torch.randn(2, 16, 11, 11)
    context = torch.randn(2, 8, 11, 11)

    with torch.no_grad():
        out, attention = module(query, context, return_attention=True)

    assert out.shape == query.shape
    assert attention.shape == (2, 48, 48)
    assert torch.isfinite(out).all()
    assert torch.isfinite(attention).all()


def test_pca_spiral_block_restores_feature_map_shape():
    module = PCAHilbertVolumeBlock(embed_dim=16)
    module.eval()
    x = torch.randn(2, 16, 11, 11)

    with torch.no_grad():
        features = module(x)

    assert hasattr(module, "output_project")
    assert (11, 11) in module._index_cache
    assert features.shape == x.shape
    assert torch.isfinite(features).all()


def test_aux_spatial_mamba_2d_block_restores_feature_map_shape():
    module = AuxSpatialMamba2DBlock(embed_dim=16)
    module.eval()
    x = torch.randn(2, 16, 11, 11)

    with torch.no_grad():
        features = module(x)

    assert hasattr(module, "hilbert_mixer")
    assert hasattr(module, "_index_cache")
    assert hasattr(module, "output_project")
    assert features.shape == x.shape
    assert torch.isfinite(features).all()


def test_pcax_stage_keeps_two_branch_feature_shapes():
    module = PCAXStage(embed_dim=16)
    module.eval()
    pca_spatial = torch.randn(2, 16, 11, 11)
    aux_spatial = torch.randn(2, 16, 11, 11)

    with torch.no_grad():
        pca_out, aux_out = module(pca_spatial, aux_spatial)

    assert isinstance(module.pca_model, PCAHilbertVolumeBlock)
    assert hasattr(module, "aux_model")
    assert isinstance(module.pca_to_aux, CovCrossAttention2d)
    assert isinstance(module.aux_to_pca, CovCrossAttention2d)
    assert torch.allclose(module.aux_to_pca_scale, torch.tensor(0.5))
    assert torch.allclose(module.pca_to_aux_scale, torch.tensor(0.5))
    assert not hasattr(module, "latent_project")
    assert not hasattr(module, "latent_transformer")
    assert pca_out.shape == pca_spatial.shape
    assert aux_out.shape == aux_spatial.shape
    assert torch.isfinite(pca_out).all()
    assert torch.isfinite(aux_out).all()


def test_ssfuse_mamba_forward_uses_current_pca_aux_path_only():
    model = SSFuseMamba(
        pca_channels=30,
        aux_channels=1,
        num_classes=6,
        embed_dim=16,
        stem_dim=8,
        stage_depths=(1,),
    )
    model.eval()
    hsi_pca = torch.randn(2, 30, 11, 11)
    aux = torch.randn(2, 1, 11, 11)

    with torch.no_grad():
        logits = model(hsi_pca, aux)

    assert not hasattr(model, "hsi_stem")
    assert not hasattr(model, "input_enhancement")
    assert logits.shape == (2, 6)
    assert torch.isfinite(logits).all()
