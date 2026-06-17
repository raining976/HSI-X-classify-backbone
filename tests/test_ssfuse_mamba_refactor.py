import torch

import src.models.ssfuse_mamba.model as ssfuse_model
from src.models.ssfuse_mamba.model import (
    CovarianceChannelAttention3d,
    HilbertLocalContext3d,
    HSI3DStem,
    MultiSource3DFusion,
    SSFuseMamba,
    SpectralSpatialFourierEnhancement,
    _hilbert_scan_indices_3d,
)


def disable_cuda_only_mamba_for_cpu_smoke_tests():
    ssfuse_model.Mamba = None


def test_spectral_spatial_fourier_enhancement_preserves_hsi_cube_shape_and_dtype():
    module = SpectralSpatialFourierEnhancement(cutoff=0.45)
    hsi = torch.randn(2, 1, 30, 11, 11)

    out = module(hsi)

    assert out.shape == hsi.shape
    assert out.dtype == hsi.dtype
    assert torch.isfinite(out).all()


def test_hilbert_scan_indices_cover_3d_cube_once():
    indices = _hilbert_scan_indices_3d(depth=4, height=3, width=5)

    assert len(indices) == 4 * 3 * 5
    assert sorted(indices) == list(range(4 * 3 * 5))


def test_hsi_stem_builds_3d_feature_volume():
    module = HSI3DStem(embed_dim=16, stem_dim=8, spec_tokens=8)
    module.eval()
    hsi = torch.randn(2, 1, 30, 11, 11)

    with torch.no_grad():
        features = module(hsi)

    assert features.shape == (2, 16, 8, 11, 11)
    assert torch.isfinite(features).all()


def test_covariance_channel_attention_preserves_3d_feature_shape():
    module = CovarianceChannelAttention3d(in_channels=12)
    module.eval()
    features = torch.randn(2, 12, 4, 5, 5)

    with torch.no_grad():
        out = module(features)

    assert out.shape == features.shape
    assert torch.isfinite(out).all()


def test_hilbert_local_context_preserves_3d_feature_shape():
    module = HilbertLocalContext3d(embed_dim=8)
    module.eval()
    features = torch.randn(2, 8, 4, 5, 5)

    with torch.no_grad():
        out = module(features)

    assert out.shape == features.shape
    assert torch.isfinite(out).all()


def test_multi_source_fusion_uses_learnable_source_weights():
    module = MultiSource3DFusion(hsi_dim=16, side_dim=8, embed_dim=8)
    module.eval()
    hsi_3d = torch.randn(2, 16, 8, 11, 11)
    pca_spatial = torch.randn(2, 8, 11, 11)
    aux_spatial = torch.randn(2, 8, 11, 11)

    with torch.no_grad():
        fused = module(hsi_3d, pca_spatial, aux_spatial)

    weights = torch.softmax(module.source_logits, dim=0)
    assert module.source_logits.requires_grad
    assert hasattr(module, "covariance_attention")
    assert hasattr(module, "hilbert_local")
    assert torch.allclose(weights, torch.tensor([0.4, 0.3, 0.3]))
    assert weights[0] > weights[1]
    assert torch.allclose(weights[1], weights[2])
    assert fused.shape == (2, 8, 8, 11, 11)
    assert torch.isfinite(fused).all()


def test_ssfuse_mamba_ablation_removes_input_fourier_enhancement():
    model = SSFuseMamba(
        hsi_channels=30,
        pca_channels=30,
        aux_channels=1,
        num_classes=6,
        embed_dim=16,
        stem_dim=8,
        spec_tokens=8,
        stage_depths=(1, 1),
    )

    assert not hasattr(model, "hsi_fourier")


def test_pca_branch_keeps_spatial_fusion_without_token_fusion():
    model = SSFuseMamba(
        hsi_channels=30,
        pca_channels=30,
        aux_channels=1,
        num_classes=6,
        embed_dim=16,
        stem_dim=8,
        spec_tokens=8,
        stage_depths=(1, 1),
    )

    assert hasattr(model, "source_fusion")
    assert not hasattr(model, "token_fusion")
    assert not hasattr(model.pca_stem, "token_project")


def test_model_uses_fused_3d_hilbert_mamba_without_2d_stages():
    model = SSFuseMamba(
        hsi_channels=30,
        pca_channels=30,
        aux_channels=1,
        num_classes=6,
        embed_dim=16,
        stem_dim=8,
        spec_tokens=8,
        stage_depths=(1, 1),
    )

    assert hasattr(model, "hilbert_mamba")
    assert not hasattr(model, "stages")
    assert not hasattr(model, "downsamples")


def test_ssfuse_mamba_forward_without_token_or_aux_fft():
    disable_cuda_only_mamba_for_cpu_smoke_tests()
    model = SSFuseMamba(
        hsi_channels=30,
        pca_channels=30,
        aux_channels=1,
        num_classes=6,
        embed_dim=16,
        stem_dim=8,
        spec_tokens=8,
        stage_depths=(1, 1),
    )
    model.eval()
    hsi = torch.randn(2, 1, 30, 11, 11)
    hsi_pca = torch.randn(2, 30, 11, 11)
    aux = torch.randn(2, 1, 11, 11)

    with torch.no_grad():
        logits = model(hsi, hsi_pca, aux)

    assert logits.shape == (2, 6)
    assert torch.isfinite(logits).all()


def test_ssfuse_mamba_forward_uses_hsi_pca_branch():
    disable_cuda_only_mamba_for_cpu_smoke_tests()
    torch.manual_seed(0)
    model = SSFuseMamba(
        hsi_channels=30,
        pca_channels=30,
        aux_channels=1,
        num_classes=6,
        embed_dim=16,
        stem_dim=8,
        spec_tokens=8,
        stage_depths=(1, 1),
    )
    model.eval()
    hsi = torch.randn(2, 1, 30, 11, 11)
    aux = torch.randn(2, 1, 11, 11)
    hsi_pca_a = torch.randn(2, 30, 11, 11)
    hsi_pca_b = hsi_pca_a + 1.0

    with torch.no_grad():
        logits_a = model(hsi, hsi_pca_a, aux)
        logits_b = model(hsi, hsi_pca_b, aux)

    assert not torch.allclose(logits_a, logits_b)
