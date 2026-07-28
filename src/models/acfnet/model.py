import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

from .HilbertScan3DMambaBlock import Hilbert3d


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


def cross_attention_weights(logits, attention_mode):
    if attention_mode == "softmax":
        return torch.softmax(logits, dim=-1)
    if attention_mode != "mutual_consistency":
        raise ValueError(f"Unsupported attention mode: {attention_mode}")

    original_dtype = logits.dtype
    work_logits = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    row_attention = torch.softmax(work_logits, dim=-1)
    column_attention = torch.softmax(work_logits, dim=-2)
    mutual = torch.sqrt(
        (row_attention * column_attention).clamp_min(torch.finfo(work_logits.dtype).tiny)
    )
    row_mass = mutual.sum(dim=-1, keepdim=True)
    normalized_mutual = mutual / row_mass.clamp_min(torch.finfo(work_logits.dtype).eps)
    shared_confidence = row_mass.clamp(max=1.0)
    return (normalized_mutual * shared_confidence).to(original_dtype)


def target_concentration_confidence(attention):
    """Summarize how confidently each target token selects its sources."""
    if attention.ndim != 3:
        raise ValueError(
            f"Expected attention with shape [B,T,S], got {tuple(attention.shape)}"
        )

    original_dtype = attention.dtype
    work_attention = (
        attention.float()
        if original_dtype in (torch.float16, torch.bfloat16)
        else attention
    )
    target_mass = work_attention.sum(dim=-1)
    target_energy = work_attention.square().sum(dim=-1)
    eps = torch.finfo(work_attention.dtype).eps
    confidence = target_energy / target_mass.clamp_min(eps)
    return confidence.clamp(0.0, 1.0).to(original_dtype)


class SpatialCrossEnhance(nn.Module):
    def __init__(self, channels, attention_mode="mutual_consistency"):
        super().__init__()
        if attention_mode not in {"softmax", "mutual_consistency"}:
            raise ValueError(f"Unsupported attention mode: {attention_mode}")
        self.attention_mode = attention_mode
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.apply_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.feedback_v_proj = nn.Conv2d(
            channels, channels, kernel_size=1, bias=False
        )
        self.feedback_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def attention(self, pca_feat, x_feat):
        _, c, _, _ = pca_feat.shape
        q = self.q_proj(pca_feat).flatten(2).transpose(1, 2)
        k = self.k_proj(x_feat).flatten(2).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(c)
        return cross_attention_weights(logits, self.attention_mode)

    def apply(self, attn, value_feat):
        b, c, h, w = value_feat.shape
        v = self.v_proj(value_feat).flatten(2).transpose(1, 2)
        enhanced = torch.matmul(attn, v).transpose(1, 2).reshape(b, c, h, w)
        return self.apply_proj(enhanced)

    def extract_feedback(self, attn, fusion_feat):
        b, c, h, w = fusion_feat.shape
        value = self.feedback_v_proj(fusion_feat).flatten(2).transpose(1, 2)
        feedback = torch.matmul(attn, value).transpose(1, 2).reshape(b, c, h, w)
        return self.feedback_proj(feedback)

    def forward(self, pca_feat, x_feat):
        attn = self.attention(pca_feat, x_feat)
        return self.apply(attn, x_feat), attn


class ChannelCrossEnhance(nn.Module):
    def __init__(self, channels, attention_mode="mutual_consistency"):
        super().__init__()
        if attention_mode not in {"softmax", "mutual_consistency"}:
            raise ValueError(f"Unsupported attention mode: {attention_mode}")
        self.attention_mode = attention_mode
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.apply_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.feedback_v_proj = nn.Conv2d(
            channels, channels, kernel_size=1, bias=False
        )
        self.feedback_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def attention(self, x_feat, pca_feat):
        _, _, h, w = x_feat.shape
        n = h * w
        q = self.q_proj(x_feat).flatten(2)
        k = self.k_proj(pca_feat).flatten(2)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(n)
        return cross_attention_weights(logits, self.attention_mode)

    def apply(self, attn, value_feat):
        b, c, h, w = value_feat.shape
        v = self.v_proj(value_feat).flatten(2)
        enhanced = torch.matmul(attn, v).reshape(b, c, h, w)
        return self.apply_proj(enhanced)

    def extract_feedback(self, attn, fusion_feat):
        b, c, h, w = fusion_feat.shape
        value = self.feedback_v_proj(fusion_feat).flatten(2)
        feedback = torch.matmul(attn, value).reshape(b, c, h, w)
        return self.feedback_proj(feedback)

    def forward(self, x_feat, pca_feat):
        attn = self.attention(x_feat, pca_feat)
        return self.apply(attn, pca_feat), attn


def generalized_hilbert_scan_indices_3d(depth, height, width):
    """Return flat CDHW indices from a generalized Hilbert cuboid traversal."""
    if min(depth, height, width) < 1:
        raise ValueError(f"Volume dimensions must be positive, got {(depth, height, width)}")

    indices = [
        (z * height + y) * width + x
        for x, y, z in Hilbert3d(width, height, depth)
    ]
    expected_length = depth * height * width
    if len(indices) != expected_length or len(set(indices)) != expected_length:
        raise RuntimeError(
            f"Generalized Hilbert traversal is invalid for {(depth, height, width)}"
        )
    return indices


class GeneralizedHilbert3DScanner(nn.Module):
    """Reversibly scan a BCHW cuboid using the generalized Hilbert traversal."""

    def __init__(self):
        super().__init__()
        self._index_cache = {}

    def scan_index(self, depth, height, width, device):
        device_key = (device.type, device.index)
        key = (depth, height, width, device_key)
        index = self._index_cache.get(key)
        if index is None:
            index = torch.tensor(
                generalized_hilbert_scan_indices_3d(depth, height, width),
                dtype=torch.long,
                device=device,
            )
            self._index_cache[key] = index
        return index

    def scan(self, feature):
        if feature.ndim != 4:
            raise ValueError(f"Expected BCHW feature map, got shape {tuple(feature.shape)}")
        b, c, h, w = feature.shape
        index = self.scan_index(c, h, w, feature.device)
        return feature.reshape(b, c * h * w).index_select(1, index).unsqueeze(-1)

    def restore(self, sequence, feature_shape):
        if len(feature_shape) != 4:
            raise ValueError(f"Expected BCHW feature shape, got {feature_shape}")
        b, c, h, w = feature_shape
        expected_length = c * h * w
        if sequence.shape != (b, expected_length, 1):
            raise ValueError(
                f"Expected sequence shape {(b, expected_length, 1)}, got {tuple(sequence.shape)}"
            )

        index = self.scan_index(c, h, w, sequence.device)
        restored = sequence.new_empty(b, expected_length)
        restored.scatter_(1, index.view(1, -1).expand(b, -1), sequence.squeeze(-1))
        return restored.view(b, c, h, w)


class PCAHilbert3DStem(nn.Module):
    def __init__(self, in_channels, hidden_dim, d_state=16):
        super().__init__()
        self.scanner = GeneralizedHilbert3DScanner()
        self.input_proj = nn.Linear(1, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.mixer = Mamba(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=4,
            expand=2,
        )
        self.output_proj = nn.Linear(hidden_dim, 1)
        self.output_block = ConvBlock(in_channels, hidden_dim)

    def forward(self, x):
        sequence = self.scanner.scan(x)
        tokens = self.input_proj(sequence)
        delta = self.output_proj(self.mixer(self.norm(tokens)))
        enhanced = x + self.scanner.restore(delta, x.shape)
        return self.output_block(enhanced)


class RowColumnMamba2DStem(nn.Module):
    def __init__(self, in_channels, hidden_dim, d_state=16):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_dim)

        self.row_forward_norm = nn.LayerNorm(hidden_dim)
        self.column_forward_norm = nn.LayerNorm(hidden_dim)
        self.row_forward_mixer = Mamba(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=4,
            expand=2,
        )
        self.column_forward_mixer = Mamba(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=4,
            expand=2,
        )

        self.direction_fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        self.output_block = ConvBlock(hidden_dim, hidden_dim)

    @staticmethod
    def _restore_rows(tokens, batch, height, width, channels):
        return tokens.reshape(batch, height, width, channels).permute(0, 3, 1, 2)

    @staticmethod
    def _restore_columns(tokens, batch, height, width, channels):
        return tokens.reshape(batch, width, height, channels).permute(0, 3, 2, 1)

    def forward(self, x):
        b, _, h, w = x.shape
        projected = self.input_proj(x.permute(0, 2, 3, 1))
        projected = projected.permute(0, 3, 1, 2)
        c = projected.shape[1]

        row_tokens = projected.permute(0, 2, 3, 1).reshape(b * h, w, c)
        row_forward = self.row_forward_mixer(self.row_forward_norm(row_tokens))
        row_forward = self._restore_rows(row_forward, b, h, w, c)

        column_tokens = projected.permute(0, 3, 2, 1).reshape(b * w, h, c)
        column_forward = self.column_forward_mixer(self.column_forward_norm(column_tokens))
        column_forward = self._restore_columns(column_forward, b, h, w, c)

        directional = self.direction_fusion(torch.cat([row_forward, column_forward], dim=1))
        return self.output_block(projected + directional)


class FusionMamba2D(nn.Module):
    def __init__(self, channels, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.mixer = Mamba(
            d_model=channels,
            d_state=d_state,
            d_conv=4,
            expand=2,
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens + self.mixer(self.norm(tokens))
        mixed = tokens.transpose(1, 2).reshape(b, c, h, w)
        return mixed + self.out_proj(mixed)


class FusionHilbertMamba3D(nn.Module):
    """Model a CxHxW fusion volume using a reversible Hilbert-3D scan."""

    def __init__(self, channels, d_state=16):
        super().__init__()
        self.scanner = GeneralizedHilbert3DScanner()
        self.input_proj = nn.Linear(1, channels)
        self.norm = nn.LayerNorm(channels)
        self.mixer = Mamba(
            d_model=channels,
            d_state=d_state,
            d_conv=4,
            expand=2,
        )
        self.output_proj = nn.Linear(channels, 1)
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        sequence = self.scanner.scan(x)
        tokens = self.input_proj(sequence)
        delta = self.output_proj(self.mixer(self.norm(tokens)))
        mixed = x + self.scanner.restore(delta, x.shape)
        return mixed + self.out_proj(mixed)


def build_fusion_mamba(fusion_scan, channels, d_state):
    if fusion_scan == "hilbert3d":
        return FusionHilbertMamba3D(channels, d_state=d_state)
    if fusion_scan == "raster2d":
        return FusionMamba2D(channels, d_state=d_state)
    raise ValueError(f"Unsupported fusion scan mode: {fusion_scan}")


class CrossModalFusionBlock(nn.Module):
    def __init__(
        self,
        channels,
        attention_mode,
        fusion_scan="hilbert3d",
        d_state=16,
        use_concentration=True,
    ):
        super().__init__()
        if not isinstance(use_concentration, bool):
            raise ValueError(
                f"use_concentration must be a boolean, got {use_concentration}"
            )
        self.pca_enhance = SpatialCrossEnhance(channels, attention_mode)
        self.x_enhance = ChannelCrossEnhance(channels, attention_mode)
        self.coarse_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.fusion_mamba = build_fusion_mamba(fusion_scan, channels, d_state)
        self.pca_res_weight = nn.Parameter(torch.tensor(-2.0))
        self.x_res_weight = nn.Parameter(torch.tensor(-2.0))
        self.fusion_scan = fusion_scan
        self.use_concentration = use_concentration

    def _joint_context(self, pca_feat, x_feat):
        pca_cross, spatial_attention = self.pca_enhance(pca_feat, x_feat)
        x_cross, channel_attention = self.x_enhance(x_feat, pca_feat)
        pca_temporary = pca_feat + pca_cross
        x_temporary = x_feat + x_cross
        coarse_feature = self.coarse_fusion(
            torch.cat([pca_temporary, x_temporary], dim=1)
        )
        joint_feature = self.fusion_mamba(coarse_feature)
        return joint_feature, spatial_attention, channel_attention

    def forward(self, pca_feat, x_feat):
        joint_feature, spatial_attention, channel_attention = self._joint_context(
            pca_feat, x_feat
        )
        pca_feedback = self.pca_enhance.extract_feedback(
            spatial_attention, joint_feature
        )
        x_feedback = self.x_enhance.extract_feedback(
            channel_attention, joint_feature
        )

        if self.use_concentration:
            b, _, h, w = pca_feedback.shape
            pca_confidence = target_concentration_confidence(
                spatial_attention
            ).reshape(b, 1, h, w)
            x_confidence = target_concentration_confidence(
                channel_attention
            ).reshape(b, x_feedback.shape[1], 1, 1)
            pca_feedback = pca_feedback * pca_confidence
            x_feedback = x_feedback * x_confidence

        pca_next = pca_feat + torch.sigmoid(self.pca_res_weight) * pca_feedback
        x_next = x_feat + torch.sigmoid(self.x_res_weight) * x_feedback
        return pca_next, x_next


class CrossModalInteractionStage(nn.Module):
    def __init__(
        self,
        pca_in_channels,
        x_in_channels,
        channels,
        attention_mode,
        fusion_scan="hilbert3d",
        d_state=16,
        use_concentration=True,
    ):
        super().__init__()
        self.pca_stem = PCAHilbert3DStem(
            pca_in_channels, channels, d_state=d_state
        )
        self.x_stem = RowColumnMamba2DStem(
            x_in_channels, channels, d_state=d_state
        )
        self.fusion = CrossModalFusionBlock(
            channels=channels,
            attention_mode=attention_mode,
            fusion_scan=fusion_scan,
            d_state=d_state,
            use_concentration=use_concentration,
        )

    def forward(self, pca_state, x_state):
        pca_feat = self.pca_stem(pca_state)
        x_feat = self.x_stem(x_state)
        return self.fusion(pca_feat, x_feat)


class ACFNet(nn.Module):
    def __init__(
        self,
        pca_channels,
        aux_channels,
        num_classes,
        hidden_dim=64,
        fusion_scan="hilbert3d",
        attention_mode="mutual_consistency",
        num_interaction_layers=2,
        d_state=16,
        use_concentration=True,
    ):
        super().__init__()
        if (
            isinstance(num_interaction_layers, bool)
            or not isinstance(num_interaction_layers, int)
            or num_interaction_layers < 1
        ):
            raise ValueError(
                f"num_interaction_layers must be a positive integer, got {num_interaction_layers}"
            )
        if isinstance(d_state, bool) or not isinstance(d_state, int) or d_state < 1:
            raise ValueError(f"d_state must be a positive integer, got {d_state}")
        if not isinstance(use_concentration, bool):
            raise ValueError(
                f"use_concentration must be a boolean, got {use_concentration}"
            )
        interaction_layers = [
            CrossModalInteractionStage(
                pca_in_channels=pca_channels,
                x_in_channels=aux_channels,
                channels=hidden_dim,
                attention_mode=attention_mode,
                fusion_scan=fusion_scan,
                d_state=d_state,
                use_concentration=use_concentration,
            )
        ]
        interaction_layers.extend(
            CrossModalInteractionStage(
                pca_in_channels=hidden_dim,
                x_in_channels=hidden_dim,
                channels=hidden_dim,
                attention_mode=attention_mode,
                fusion_scan=fusion_scan,
                d_state=d_state,
                use_concentration=use_concentration,
            )
            for _ in range(num_interaction_layers - 1)
        )
        self.interaction_layers = nn.ModuleList(interaction_layers)
        self.final_fusion_proj = nn.Sequential(
            nn.Conv2d(
                hidden_dim * 2,
                hidden_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        self.final_fusion_mamba = build_fusion_mamba(
            fusion_scan, hidden_dim, d_state
        )
        self.fusion_scan = fusion_scan
        self.attention_mode = attention_mode
        self.num_interaction_layers = num_interaction_layers
        self.d_state = d_state
        self.use_concentration = use_concentration
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, hsi_pca, aux):
        pca_state = hsi_pca
        x_state = aux
        for interaction_layer in self.interaction_layers:
            pca_state, x_state = interaction_layer(pca_state, x_state)

        fusion_feat = self.final_fusion_proj(
            torch.cat([pca_state, x_state], dim=1)
        )
        fusion_feat = self.final_fusion_mamba(fusion_feat)
        return self.classifier(fusion_feat)
