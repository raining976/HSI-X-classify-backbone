import pytest
import torch
import torch.nn as nn

import src.models.acfnet.model as acfnet_model
from src.models.acfnet.adapter import build_model
from src.models.acfnet.model import (
    ACFNet,
    CrossModalFusionBlock,
    CrossModalInteractionStage,
    FusionHilbertMamba3D,
    FusionMamba2D,
    GeneralizedHilbert3DScanner,
    PCAHilbert3DStem,
    RowColumnMamba2DStem,
    cross_attention_weights,
    generalized_hilbert_scan_indices_3d,
    target_concentration_confidence,
)


class CPUMambaStub(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.init_kwargs = kwargs
        self.input_shapes = []

    def forward(self, tokens):
        self.input_shapes.append(tuple(tokens.shape))
        return tokens


@pytest.fixture(autouse=True)
def use_cpu_mamba_stub(monkeypatch):
    monkeypatch.setattr(acfnet_model, "Mamba", CPUMambaStub)


def test_softmax_attention_matches_torch_softmax():
    logits = torch.randn(2, 5, 5)

    attention = cross_attention_weights(logits, "softmax")

    assert torch.equal(attention, torch.softmax(logits, dim=-1))


def test_mutual_consistency_attention_is_finite_and_bounded():
    attention = cross_attention_weights(torch.randn(2, 7, 7), "mutual_consistency")

    assert torch.isfinite(attention).all()
    assert torch.all(attention >= 0)
    assert torch.all(attention.sum(dim=-1) <= 1.0 + 1e-6)


def test_target_concentration_distinguishes_identity_and_uniform_attention():
    size = 7
    identity = torch.eye(size).unsqueeze(0)
    uniform = torch.full((1, size, size), 1.0 / size)

    identity_confidence = target_concentration_confidence(identity)
    uniform_confidence = target_concentration_confidence(uniform)

    assert torch.allclose(identity_confidence, torch.ones_like(identity_confidence))
    assert torch.allclose(
        uniform_confidence, torch.full_like(uniform_confidence, 1.0 / size)
    )


def test_target_concentration_handles_zero_rows_and_low_precision():
    attention = torch.tensor(
        [[[1.0, 0.0], [0.0, 0.0]]], dtype=torch.float16
    )

    confidence = target_concentration_confidence(attention)

    assert confidence.dtype == attention.dtype
    assert torch.equal(confidence, torch.tensor([[1.0, 0.0]], dtype=torch.float16))
    assert torch.isfinite(confidence).all()


def test_generalized_hilbert_indices_and_scanner_are_reversible():
    depth, height, width = 4, 3, 5
    indices = generalized_hilbert_scan_indices_3d(depth, height, width)
    scanner = GeneralizedHilbert3DScanner()
    feature = torch.arange(2 * depth * height * width, dtype=torch.float32).reshape(
        2, depth, height, width
    )

    restored = scanner.restore(scanner.scan(feature), feature.shape)

    assert len(indices) == depth * height * width
    assert sorted(indices) == list(range(depth * height * width))
    assert torch.equal(restored, feature)


def test_two_stems_preserve_shape_and_receive_d_state():
    pca_stem = PCAHilbert3DStem(6, 8, d_state=12)
    x_stem = RowColumnMamba2DStem(3, 8, d_state=12)

    pca_output = pca_stem(torch.randn(2, 6, 5, 7))
    x_output = x_stem(torch.randn(2, 3, 5, 7))

    assert pca_output.shape == (2, 8, 5, 7)
    assert x_output.shape == (2, 8, 5, 7)
    assert pca_stem.mixer.init_kwargs["d_state"] == 12
    assert x_stem.row_forward_mixer.init_kwargs["d_state"] == 12
    assert x_stem.column_forward_mixer.init_kwargs["d_state"] == 12


class AttentionStub(nn.Module):
    def __init__(self, cross_value, feedback_value, attention_kind):
        super().__init__()
        self.cross_value = cross_value
        self.feedback_value = feedback_value
        self.attention_kind = attention_kind
        self.feedback_calls = 0

    def forward(self, target, _source):
        size = target.shape[-2] * target.shape[-1]
        if self.attention_kind == "channel":
            size = target.shape[1]
        attention = torch.eye(size, device=target.device).unsqueeze(0)
        attention = attention.expand(target.shape[0], -1, -1)
        return torch.full_like(target, self.cross_value), attention

    def extract_feedback(self, attention, joint_feature):
        self.feedback_calls += 1
        self.feedback_attention = attention
        self.feedback_joint_ptr = joint_feature.data_ptr()
        return torch.full_like(joint_feature, self.feedback_value)


class CoarseCapture(nn.Module):
    def forward(self, concatenated):
        self.input = concatenated.detach().clone()
        return concatenated[:, : concatenated.shape[1] // 2]


def make_fusion_block_for_flow_test(use_concentration=True):
    block = CrossModalFusionBlock(
        channels=2,
        attention_mode="softmax",
        use_concentration=use_concentration,
    )
    block.pca_enhance = AttentionStub(2.0, 5.0, "spatial")
    block.x_enhance = AttentionStub(3.0, 7.0, "channel")
    block.coarse_fusion = CoarseCapture()
    block.fusion_mamba = nn.Identity()
    return block


def test_fusion_block_uses_temporary_cross_residuals_only_for_coarse_fusion():
    block = make_fusion_block_for_flow_test()
    pca = torch.ones(1, 2, 2, 2)
    x = torch.full_like(pca, 4.0)

    pca_next, x_next = block(pca, x)
    residual_weight = torch.sigmoid(torch.tensor(-2.0))

    assert torch.equal(
        block.coarse_fusion.input[:, :2], torch.full_like(pca, 3.0)
    )
    assert torch.equal(
        block.coarse_fusion.input[:, 2:], torch.full_like(x, 7.0)
    )
    assert torch.allclose(pca_next, pca + residual_weight * 5.0)
    assert torch.allclose(x_next, x + residual_weight * 7.0)


def test_feedback_reuses_original_attention_directions_without_transpose():
    block = make_fusion_block_for_flow_test()

    block(torch.ones(1, 2, 2, 2), torch.ones(1, 2, 2, 2))

    assert block.pca_enhance.feedback_calls == 1
    assert block.x_enhance.feedback_calls == 1
    assert block.pca_enhance.feedback_attention.shape == (1, 4, 4)
    assert block.x_enhance.feedback_attention.shape == (1, 2, 2)
    assert block.pca_enhance.feedback_joint_ptr == block.x_enhance.feedback_joint_ptr


def test_disabling_concentration_keeps_scalar_gated_feedback():
    block = make_fusion_block_for_flow_test(use_concentration=False)
    pca = torch.ones(1, 2, 2, 2)
    x = torch.ones_like(pca)

    pca_next, x_next = block(pca, x)
    residual_weight = torch.sigmoid(torch.tensor(-2.0))

    assert torch.allclose(pca_next, pca + residual_weight * 5.0)
    assert torch.allclose(x_next, x + residual_weight * 7.0)


def test_feedback_projections_are_independent_from_cross_value_projections():
    block = CrossModalFusionBlock(8, "softmax")

    assert block.pca_enhance.feedback_v_proj is not block.pca_enhance.v_proj
    assert block.x_enhance.feedback_v_proj is not block.x_enhance.v_proj
    assert block.pca_enhance.feedback_proj is not block.pca_enhance.apply_proj
    assert block.x_enhance.feedback_proj is not block.x_enhance.apply_proj


@pytest.mark.parametrize("fusion_scan", ["hilbert3d", "raster2d"])
def test_stage_contains_stems_and_its_own_fusion_mamba(fusion_scan):
    stage = CrossModalInteractionStage(
        6, 1, 8, "softmax", fusion_scan=fusion_scan, d_state=12
    )

    pca_output, x_output = stage(
        torch.randn(2, 6, 5, 5), torch.randn(2, 1, 5, 5)
    )

    expected_type = FusionHilbertMamba3D if fusion_scan == "hilbert3d" else FusionMamba2D
    assert isinstance(stage.pca_stem, PCAHilbert3DStem)
    assert isinstance(stage.x_stem, RowColumnMamba2DStem)
    assert isinstance(stage.fusion, CrossModalFusionBlock)
    assert isinstance(stage.fusion.fusion_mamba, expected_type)
    assert pca_output.shape == (2, 8, 5, 5)
    assert x_output.shape == (2, 8, 5, 5)


def test_stacked_stages_and_final_fusion_mamba_have_independent_parameters():
    model = ACFNet(6, 1, 4, hidden_dim=8, num_interaction_layers=2)
    first, second = model.interaction_layers

    assert first.pca_stem is not second.pca_stem
    assert first.x_stem is not second.x_stem
    assert first.fusion.pca_enhance is not second.fusion.pca_enhance
    assert first.fusion.fusion_mamba is not second.fusion.fusion_mamba
    assert model.final_fusion_mamba is not first.fusion.fusion_mamba
    assert model.final_fusion_mamba is not second.fusion.fusion_mamba


def test_final_fusion_receives_complete_enhanced_branch_states():
    class StageStub(nn.Module):
        def forward(self, pca, x):
            return pca + 2.0, x + 3.0

    model = ACFNet(8, 8, 4, hidden_dim=8, num_interaction_layers=1)
    model.interaction_layers = nn.ModuleList([StageStub()])
    model.final_fusion_mamba = nn.Identity()
    captured = {}

    def capture_projection_input(_module, inputs):
        captured["input"] = inputs[0].detach().clone()

    handle = model.final_fusion_proj.register_forward_pre_hook(capture_projection_input)
    logits = model(torch.ones(2, 8, 5, 5), torch.ones(2, 8, 5, 5))
    handle.remove()

    assert captured["input"].shape == (2, 16, 5, 5)
    assert torch.equal(captured["input"][:, :8], torch.full((2, 8, 5, 5), 3.0))
    assert torch.equal(captured["input"][:, 8:], torch.full((2, 8, 5, 5), 4.0))
    assert logits.shape == (2, 4)


def test_adapter_passes_fusion_scan_and_removes_old_modes():
    class ConfigStub:
        values = {
            "channels": 6,
            "lidar_or_sar_channels": [1],
            "out_features": [4],
            "acfnet_attention_mode": "softmax",
            "acfnet_num_interaction_layers": 3,
            "acfnet_fusion_scan": "raster2d",
            "acfnet_d_state": 24,
            "acfnet_use_concentration": False,
        }

        def get_value(self, key):
            return self.values.get(key)

    model = build_model(ConfigStub(), dataset_type=0, device="cpu")["net"]

    assert model.fusion_scan == "raster2d"
    assert model.attention_mode == "softmax"
    assert model.num_interaction_layers == 3
    assert model.d_state == 24
    assert model.use_concentration is False
    assert all(
        isinstance(stage.fusion.fusion_mamba, FusionMamba2D)
        for stage in model.interaction_layers
    )
    assert isinstance(model.final_fusion_mamba, FusionMamba2D)
    assert not hasattr(model, "enhance_mode")
    assert not hasattr(model, "final_fusion")


@pytest.mark.parametrize("fusion_scan", ["hilbert3d", "raster2d"])
@pytest.mark.parametrize("attention_mode", ["softmax", "mutual_consistency"])
@pytest.mark.parametrize("use_concentration", [True, False])
def test_acfnet_forward_backward_combinations(
    fusion_scan, attention_mode, use_concentration
):
    model = ACFNet(
        6,
        1,
        4,
        hidden_dim=8,
        fusion_scan=fusion_scan,
        attention_mode=attention_mode,
        use_concentration=use_concentration,
    )
    pca = torch.randn(2, 6, 5, 5, requires_grad=True)
    x = torch.randn(2, 1, 5, 5, requires_grad=True)

    logits = model(pca, x)
    logits.mean().backward()

    assert logits.shape == (2, 4)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(pca.grad).all()
    assert torch.isfinite(x.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


@pytest.mark.parametrize("num_layers", [1, 2, 3])
def test_configurable_stage_depth_and_linear_parameter_growth(num_layers):
    model = ACFNet(6, 1, 4, hidden_dim=8, num_interaction_layers=num_layers)

    assert len(model.interaction_layers) == num_layers


def test_parameter_count_grows_linearly_with_independent_stages():
    counts = [
        sum(p.numel() for p in ACFNet(6, 1, 4, hidden_dim=8, num_interaction_layers=n).parameters())
        for n in (1, 2, 3)
    ]

    assert counts[1] - counts[0] == counts[2] - counts[1]


@pytest.mark.parametrize("num_layers", [0, -1, 1.5, True])
def test_invalid_interaction_layer_count_is_rejected(num_layers):
    with pytest.raises(ValueError, match="num_interaction_layers"):
        ACFNet(6, 1, 4, hidden_dim=8, num_interaction_layers=num_layers)


@pytest.mark.parametrize("d_state", [0, -1, 1.5, True])
def test_invalid_d_state_is_rejected(d_state):
    with pytest.raises(ValueError, match="d_state"):
        ACFNet(6, 1, 4, hidden_dim=8, d_state=d_state)


def test_invalid_fusion_scan_is_rejected():
    with pytest.raises(ValueError, match="fusion scan mode"):
        ACFNet(6, 1, 4, hidden_dim=8, fusion_scan="invalid")
