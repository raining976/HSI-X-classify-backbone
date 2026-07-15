import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


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


class SpatialCrossEnhance(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.apply_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.extract_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.res_weight = nn.Parameter(torch.tensor(-2.0))

    def attention(self, pca_feat, x_feat):
        _, c, _, _ = pca_feat.shape
        q = self.q_proj(pca_feat).flatten(2).transpose(1, 2)
        k = self.k_proj(x_feat).flatten(2).transpose(1, 2)
        return torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(c), dim=-1)

    def apply(self, attn, value_feat, proj):
        b, c, h, w = value_feat.shape
        v = self.v_proj(value_feat).flatten(2).transpose(1, 2)
        enhanced = torch.matmul(attn, v).transpose(1, 2).reshape(b, c, h, w)
        return proj(enhanced)

    def fusion_feature(self, pca_feat, x_feat, attn):
        return pca_feat + self.apply(attn, x_feat, self.apply_proj)

    def extract(self, fusion_feat, attn):
        return self.apply(attn, fusion_feat, self.extract_proj)

    def forward(self, pca_feat, x_feat):
        attn = self.attention(pca_feat, x_feat)
        enhanced = self.extract(x_feat, attn)
        return pca_feat + torch.sigmoid(self.res_weight) * enhanced


class ChannelCrossEnhance(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.apply_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.extract_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.res_weight = nn.Parameter(torch.tensor(-2.0))

    def attention(self, x_feat, pca_feat):
        _, _, h, w = x_feat.shape
        n = h * w
        q = self.q_proj(x_feat).flatten(2)
        k = self.k_proj(pca_feat).flatten(2)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        return torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(n), dim=-1)

    def apply(self, attn, value_feat, proj):
        b, c, h, w = value_feat.shape
        v = self.v_proj(value_feat).flatten(2)
        enhanced = torch.matmul(attn, v).reshape(b, c, h, w)
        return proj(enhanced)

    def fusion_feature(self, x_feat, pca_feat, attn):
        return x_feat + self.apply(attn, pca_feat, self.apply_proj)

    def extract(self, fusion_feat, attn):
        return self.apply(attn, fusion_feat, self.extract_proj)

    def forward(self, x_feat, pca_feat):
        attn = self.attention(x_feat, pca_feat)
        enhanced = self.extract(pca_feat, attn)
        return x_feat + torch.sigmoid(self.res_weight) * enhanced


def _hilbert_integer_from_point_3d(point, bits):
    """Map one 3D integer coordinate to its Hilbert-curve distance."""
    axes = list(point)
    if len(axes) != 3:
        raise ValueError(f"Expected a 3D point, got {len(axes)} dimensions")
    if bits < 1:
        raise ValueError(f"bits must be positive, got {bits}")

    max_coord = (1 << bits) - 1
    if any(coord < 0 or coord > max_coord for coord in axes):
        raise ValueError(f"Point {point} is outside the {bits}-bit Hilbert cube")

    # Inverse undo from John Skilling's transpose representation.
    q = 1 << (bits - 1)
    while q > 1:
        p = q - 1
        for axis in range(3):
            if axes[axis] & q:
                axes[0] ^= p
            else:
                t = (axes[0] ^ axes[axis]) & p
                axes[0] ^= t
                axes[axis] ^= t
        q >>= 1

    for axis in range(1, 3):
        axes[axis] ^= axes[axis - 1]

    t = 0
    q = 1 << (bits - 1)
    while q > 1:
        if axes[2] & q:
            t ^= q - 1
        q >>= 1
    for axis in range(3):
        axes[axis] ^= t

    distance = 0
    for bit in range(bits - 1, -1, -1):
        for axis in range(3):
            distance = (distance << 1) | ((axes[axis] >> bit) & 1)
    return distance


def hilbert_scan_indices_3d(depth, height, width):
    """Return flat CDHW indices ordered by a 3D Hilbert curve."""
    if min(depth, height, width) < 1:
        raise ValueError(f"Volume dimensions must be positive, got {(depth, height, width)}")

    # Embed a non-cubic volume in the smallest power-of-two Hilbert cube and
    # retain only valid coordinates. This keeps the scan defined for shapes
    # such as 64 x 11 x 11 without padding the feature tensor itself.
    bits = max(1, (max(depth, height, width) - 1).bit_length())
    ordered = []
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                distance = _hilbert_integer_from_point_3d((z, y, x), bits)
                flat_index = (z * height + y) * width + x
                ordered.append((distance, flat_index))
    ordered.sort(key=lambda item: item[0])
    return [flat_index for _, flat_index in ordered]


class Hilbert3DScanner(nn.Module):
    """Reversibly reorder a BCHW feature volume along a 3D Hilbert curve."""

    def __init__(self):
        super().__init__()
        self._index_cache = {}

    def scan_index(self, depth, height, width, device):
        device_key = (device.type, device.index)
        key = (depth, height, width, device_key)
        index = self._index_cache.get(key)
        if index is None:
            index = torch.tensor(
                hilbert_scan_indices_3d(depth, height, width),
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
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.scanner = Hilbert3DScanner()
        self.input_proj = nn.Linear(1, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.mixer = Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)
        self.output_proj = nn.Linear(hidden_dim, 1)
        self.output_block = ConvBlock(in_channels, hidden_dim)

    def forward(self, x):
        sequence = self.scanner.scan(x)
        tokens = self.input_proj(sequence)
        delta = self.output_proj(self.mixer(self.norm(tokens)))
        enhanced = x + self.scanner.restore(delta, x.shape)
        return self.output_block(enhanced)


class RowColumnMamba2DStem(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_dim)

        self.row_forward_norm = nn.LayerNorm(hidden_dim)
        self.column_forward_norm = nn.LayerNorm(hidden_dim)
        self.row_forward_mixer = Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)
        self.column_forward_mixer = Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)

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
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.mixer = Mamba(d_model=channels, d_state=16, d_conv=4, expand=2)
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

    def __init__(self, channels):
        super().__init__()
        self.scanner = Hilbert3DScanner()
        self.input_proj = nn.Linear(1, channels)
        self.norm = nn.LayerNorm(channels)
        self.mixer = Mamba(d_model=channels, d_state=16, d_conv=4, expand=2)
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


class ACFNet(nn.Module):
    def __init__(self, pca_channels, aux_channels, num_classes, hidden_dim=64, fusion_scan="hilbert3d"):
        super().__init__()
        self.pca_stem = PCAHilbert3DStem(pca_channels, hidden_dim)
        self.x_stem = RowColumnMamba2DStem(aux_channels, hidden_dim)

        self.pca_enhance = SpatialCrossEnhance(hidden_dim)
        self.x_enhance = ChannelCrossEnhance(hidden_dim)
        if fusion_scan == "hilbert3d":
            self.fusion_mamba = FusionHilbertMamba3D(hidden_dim)
        elif fusion_scan == "raster2d":
            self.fusion_mamba = FusionMamba2D(hidden_dim)
        else:
            raise ValueError(f"Unsupported fusion scan mode: {fusion_scan}")
        self.fusion_scan = fusion_scan

        self.pca_model = nn.Sequential(
            ConvBlock(hidden_dim, hidden_dim),
            # ConvBlock(hidden_dim, hidden_dim),
        )
        self.x_model = nn.Sequential(
            ConvBlock(hidden_dim, hidden_dim),
            # ConvBlock(hidden_dim, hidden_dim),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            ConvBlock(hidden_dim, hidden_dim),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, hsi_pca, aux):
        pca_feat = self.pca_stem(hsi_pca)
        x_feat = self.x_stem(aux)

        spatial_attn = self.pca_enhance.attention(pca_feat, x_feat)
        channel_attn = self.x_enhance.attention(x_feat, pca_feat)

        pca_fusion = self.pca_enhance.fusion_feature(pca_feat, x_feat, spatial_attn)
        x_fusion = self.x_enhance.fusion_feature(x_feat, pca_feat, channel_attn)
        fusion_feat = self.fusion_mamba(pca_fusion + x_fusion)

        pca_delta = self.pca_enhance.extract(fusion_feat, spatial_attn)
        x_delta = self.x_enhance.extract(fusion_feat, channel_attn)
        pca_enhanced = pca_feat + torch.sigmoid(self.pca_enhance.res_weight) * pca_delta
        x_enhanced = x_feat + torch.sigmoid(self.x_enhance.res_weight) * x_delta

        pca_out = self.pca_model(pca_enhanced)
        x_out = self.x_model(x_enhanced)
        fused = self.fusion(torch.cat([pca_out, x_out], dim=1))
        return self.classifier(fused)
