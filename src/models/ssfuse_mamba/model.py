import torch
import torch.nn as nn
from mamba_ssm import Mamba


def _hilbert_integer_from_point(point, bits):
    axes = list(point)
    dims = len(axes)
    if bits <= 0:
        return 0

    mask = 1 << (bits - 1)
    q = mask
    while q > 1:
        p = q - 1
        for idx in range(dims):
            if axes[idx] & q:
                axes[0] ^= p
            else:
                t = (axes[0] ^ axes[idx]) & p
                axes[0] ^= t
                axes[idx] ^= t
        q >>= 1

    for idx in range(1, dims):
        axes[idx] ^= axes[idx - 1]

    t = 0
    q = mask
    while q > 1:
        if axes[-1] & q:
            t ^= q - 1
        q >>= 1

    for idx in range(dims):
        axes[idx] ^= t

    distance = 0
    for bit in range(bits - 1, -1, -1):
        for idx in range(dims):
            distance = (distance << 1) | ((axes[idx] >> bit) & 1)
    return distance


def _hilbert_scan_indices_3d(depth, height, width):
    max_size = max(depth, height, width)
    bits = max(1, (max_size - 1).bit_length())
    coords = []
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                distance = _hilbert_integer_from_point((z, y, x), bits)
                coords.append((distance, z * height * width + y * width + x))
    coords.sort(key=lambda item: item[0])
    return [flat_idx for _, flat_idx in coords]


def _hilbert_scan_indices_2d(height, width):
    max_size = max(height, width)
    bits = max(1, (max_size - 1).bit_length())
    coords = []
    for y in range(height):
        for x in range(width):
            distance = _hilbert_integer_from_point((y, x), bits)
            coords.append((distance, y * width + x))
    coords.sort(key=lambda item: item[0])
    return [flat_idx for _, flat_idx in coords]


def _group_mean_pool_tokens(tokens, out_tokens):
    b, length, channels = tokens.shape
    if length == out_tokens:
        return tokens
    pad = (out_tokens - length % out_tokens) % out_tokens
    if pad > 0:
        tokens = torch.cat([tokens, tokens[:, -1:, :].expand(-1, pad, -1)], dim=1)
    return tokens.view(b, out_tokens, -1, channels).mean(dim=2)


def _group_mean_pool_depth(x3d, out_depth):
    b, channels, depth, height, width = x3d.shape
    if depth == out_depth:
        return x3d
    pad = (out_depth - depth % out_depth) % out_depth
    if pad > 0:
        x3d = torch.cat([x3d, x3d[:, :, -1:, :, :].expand(-1, -1, pad, -1, -1)], dim=2)
    return x3d.view(b, channels, out_depth, -1, height, width).mean(dim=3)


class ConvBNGELU3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseResidual2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.mix = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.mix(x))


class DepthwiseResidual3d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.mix = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, bias=False),
            nn.BatchNorm3d(embed_dim),
            nn.GELU(),
            nn.Conv3d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm3d(embed_dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.mix(x))


class MambaTokenMixer(nn.Module):
    def __init__(self, embed_dim, d_state=8, d_conv=3, expand=1):
        super().__init__()
        self.input_norm = nn.LayerNorm(embed_dim)
        self.ssm_norm = nn.LayerNorm(embed_dim)
        self.depthwise_conv = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=d_conv,
            padding=d_conv // 2,
            groups=embed_dim,
            bias=True,
        )
        self.ssm = Mamba(d_model=embed_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.ssm_out_norm = nn.LayerNorm(embed_dim)
        self.gate_proj = nn.Linear(embed_dim, embed_dim)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, tokens):
        shortcut = tokens
        tokens = self.input_norm(tokens)

        ssm_tokens = self.ssm_norm(tokens)
        ssm_tokens = self.depthwise_conv(ssm_tokens.transpose(1, 2)).transpose(1, 2)
        ssm_tokens = self.act(ssm_tokens)
        ssm_tokens = self.ssm(ssm_tokens)
        ssm_tokens = self.ssm_out_norm(ssm_tokens)

        gate = torch.sigmoid(self.gate_proj(tokens))
        tokens = self.out_proj(ssm_tokens * gate)
        return shortcut + tokens


class HilbertMamba3DModeling(nn.Module):
    def __init__(self, embed_dim, out_tokens):
        super().__init__()
        self.out_tokens = out_tokens
        self.input_project = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.sequence_mixer = MambaTokenMixer(embed_dim)
        self._index_cache = {}

    def _scan_index(self, depth, height, width, device):
        key = (depth, height, width)
        index = self._index_cache.get(key)
        if index is None:
            index = torch.tensor(_hilbert_scan_indices_3d(depth, height, width), dtype=torch.long)
            self._index_cache[key] = index
        return index.to(device=device)

    def forward(self, x3d):
        b, c, depth, height, width = x3d.shape
        scan_index = self._scan_index(depth, height, width, x3d.device)
        sequence = x3d.flatten(2).index_select(dim=2, index=scan_index).transpose(1, 2)
        sequence = self.sequence_mixer(self.input_project(sequence))
        return _group_mean_pool_tokens(sequence, self.out_tokens)


class PCAHilbertMamba3DModeling(nn.Module):
    def __init__(self, embed_dim, out_tokens):
        super().__init__()
        self.out_tokens = out_tokens
        self.input_project = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.sequence_mixer = MambaTokenMixer(embed_dim)
        self._index_cache = {}

    def _scan_index(self, depth, height, width, device):
        key = (depth, height, width)
        index = self._index_cache.get(key)
        if index is None:
            index = torch.tensor(_hilbert_scan_indices_3d(depth, height, width), dtype=torch.long)
            self._index_cache[key] = index
        return index.to(device=device)

    def forward(self, x2d):
        b, c, height, width = x2d.shape
        scan_index = self._scan_index(c, height, width, x2d.device)
        sequence = x2d.unsqueeze(1).flatten(2).index_select(dim=2, index=scan_index).transpose(1, 2)
        sequence = self.sequence_mixer(self.input_project(sequence))
        return sequence.mean(dim=1)


class PCAHilbertVolumeBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.input_project = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.sequence_mixer = MambaTokenMixer(embed_dim)
        self.output_project = nn.Linear(embed_dim, embed_dim)
        self._index_cache = {}

    def _scan_index(self, height, width, device):
        key = (height, width)
        index = self._index_cache.get(key)
        if index is None:
            index = torch.tensor(_hilbert_scan_indices_2d(height, width), dtype=torch.long)
            self._index_cache[key] = index
        return index.to(device=device)

    def forward(self, x2d):
        b, c, height, width = x2d.shape
        scan_index = self._scan_index(height, width, x2d.device)
        sequence = x2d.flatten(2).index_select(dim=2, index=scan_index).transpose(1, 2)
        sequence = self.sequence_mixer(self.input_project(sequence))
        sequence = self.output_project(sequence)
        restored = x2d.new_empty(b, height * width, c)
        restored.scatter_(dim=1, index=scan_index.view(1, -1, 1).expand(b, -1, c), src=sequence)
        return x2d + restored.transpose(1, 2).view(b, c, height, width)


class RasterMamba3DModeling(nn.Module):
    def __init__(self, embed_dim, out_tokens):
        super().__init__()
        self.out_tokens = out_tokens
        self.input_project = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.sequence_mixer = MambaTokenMixer(embed_dim)

    def forward(self, x3d):
        sequence = x3d.flatten(2).transpose(1, 2)
        sequence = self.sequence_mixer(self.input_project(sequence))
        return _group_mean_pool_tokens(sequence, self.out_tokens)


class PooledTokenModeling(nn.Module):
    def __init__(self, embed_dim, out_tokens):
        super().__init__()
        self.out_tokens = out_tokens
        self.token_project = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )

    def forward(self, x3d):
        tokens = _group_mean_pool_depth(x3d, self.out_tokens).mean(dim=(-1, -2)).transpose(1, 2)
        return self.token_project(tokens)


class AuxSpatialMamba2DBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.input_project = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.hilbert_mixer = MambaTokenMixer(embed_dim)
        self.output_project = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
        )
        self.act = nn.GELU()
        self._index_cache = {}

    def _scan_index(self, height, width, device):
        key = (height, width)
        index = self._index_cache.get(key)
        if index is None:
            index = torch.tensor(_hilbert_scan_indices_2d(height, width), dtype=torch.long)
            self._index_cache[key] = index
        return index.to(device=device)

    def forward(self, x):
        b, c, height, width = x.shape
        scan_index = self._scan_index(height, width, x.device)
        sequence = x.flatten(2).index_select(dim=2, index=scan_index).transpose(1, 2)
        sequence = self.hilbert_mixer(self.input_project(sequence))
        restored = x.new_empty(b, height * width, c)
        restored.scatter_(dim=1, index=scan_index.view(1, -1, 1).expand(b, -1, c), src=sequence)
        feature = restored.transpose(1, 2).view(b, c, height, width)
        return self.act(x + self.output_project(feature))


class AuxSpatialMeanPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        _ = embed_dim

    def forward(self, x):
        return x.mean(dim=(2, 3))


class CovCrossAttention3d(nn.Module):
    def __init__(self, query_channels, context_channels, attn_channels=48):
        super().__init__()
        self.attn_channels = attn_channels
        self.query_proj = nn.Conv3d(query_channels, attn_channels, kernel_size=1, bias=False)
        self.key_proj = nn.Conv3d(context_channels, attn_channels, kernel_size=1, bias=False)
        self.value_proj = nn.Conv3d(context_channels, attn_channels, kernel_size=1, bias=False)
        self.out_proj = nn.Conv3d(attn_channels, query_channels, kernel_size=1, bias=False)
        self.query_norm = nn.LayerNorm(attn_channels)
        self.key_norm = nn.LayerNorm(attn_channels)

    def forward(self, query, context, return_attention=False):
        b, _, depth, height, width = query.shape
        n = depth * height * width
        query_flat = self.query_proj(query).flatten(2)
        key_flat = self.key_proj(context).flatten(2)
        value = self.value_proj(context)
        value_flat = value.flatten(2)
        query_norm = self.query_norm(query_flat.transpose(1, 2)).transpose(1, 2)
        key_norm = self.key_norm(key_flat.transpose(1, 2)).transpose(1, 2)
        query_centered = query_norm - query_norm.mean(dim=-1, keepdim=True)
        key_centered = key_norm - key_norm.mean(dim=-1, keepdim=True)
        covariance = torch.matmul(query_centered, key_centered.transpose(1, 2)) / max(n - 1, 1)
        attention = torch.softmax(covariance, dim=-1)
        enhanced = torch.matmul(attention, value_flat).view(b, -1, depth, height, width)
        enhanced = self.out_proj(enhanced)
        if return_attention:
            return enhanced, attention
        return enhanced


class CovCrossAttention2d(nn.Module):
    def __init__(self, query_channels, context_channels, attn_channels=48):
        super().__init__()
        self.attn_channels = attn_channels
        self.query_proj = nn.Conv2d(query_channels, attn_channels, kernel_size=1, bias=False)
        self.key_proj = nn.Conv2d(context_channels, attn_channels, kernel_size=1, bias=False)
        self.value_proj = nn.Conv2d(context_channels, attn_channels, kernel_size=1, bias=False)
        self.value_spatial_mix = nn.Sequential(
            nn.Conv2d(attn_channels, attn_channels, kernel_size=3, padding=1, groups=attn_channels, bias=False),
            nn.BatchNorm2d(attn_channels),
            nn.GELU(),
        )
        self.out_proj = nn.Conv2d(attn_channels, query_channels, kernel_size=1, bias=False)
        self.spatial_context = nn.Sequential(
            nn.Conv2d(query_channels + context_channels, attn_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(attn_channels),
            nn.GELU(),
            nn.Conv2d(attn_channels, query_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(query_channels),
        )
        self.value_spatial_scale = nn.Parameter(torch.zeros(1))
        self.spatial_context_scale = nn.Parameter(torch.zeros(1))
        self.center_spatial_scale = nn.Parameter(torch.zeros(1))
        self.query_norm = nn.LayerNorm(attn_channels)
        self.key_norm = nn.LayerNorm(attn_channels)

    def forward(self, query, context, return_attention=False):
        b, _, height, width = query.shape
        n = height * width
        query_flat = self.query_proj(query).flatten(2)
        key_flat = self.key_proj(context).flatten(2)
        value = self.value_proj(context)
        value = value + self.value_spatial_scale * self.value_spatial_mix(value)
        value_flat = value.flatten(2)

        center_idx = (height // 2) * width + (width // 2)
        query_tokens = query_flat.transpose(1, 2)
        key_tokens = key_flat.transpose(1, 2)
        value_tokens = value_flat.transpose(1, 2)
        center_query = query_tokens[:, center_idx:center_idx + 1, :]
        spatial_logits = torch.matmul(center_query, key_tokens.transpose(1, 2)) * (self.attn_channels ** -0.5)
        spatial_attention = torch.softmax(spatial_logits, dim=-1)
        center_context = torch.matmul(spatial_attention, value_tokens).transpose(1, 2).view(b, -1, 1, 1)

        query_norm = self.query_norm(query_flat.transpose(1, 2)).transpose(1, 2)
        key_norm = self.key_norm(key_flat.transpose(1, 2)).transpose(1, 2)
        query_centered = query_norm - query_norm.mean(dim=-1, keepdim=True)
        key_centered = key_norm - key_norm.mean(dim=-1, keepdim=True)
        covariance = torch.matmul(query_centered, key_centered.transpose(1, 2)) / max(n - 1, 1)
        attention = torch.softmax(covariance, dim=-1)
        enhanced = torch.matmul(attention, value_flat).view(b, -1, height, width)
        enhanced = enhanced + self.center_spatial_scale * center_context.expand(-1, -1, height, width)
        enhanced = self.out_proj(enhanced)
        enhanced = enhanced + self.spatial_context_scale * self.spatial_context(torch.cat([query, context], dim=1))
        if return_attention:
            return enhanced, attention
        return enhanced


class HSI3DStem(nn.Module):
    def __init__(self, embed_dim, stem_dim=24, spec_tokens=16):
        super().__init__()
        self.spec_tokens = spec_tokens
        out_dim = stem_dim
        self.input_project = ConvBNGELU3d(1, out_dim, kernel_size=(3, 3, 3), padding=1)
        self.residual = DepthwiseResidual3d(out_dim)
        _ = embed_dim

    def forward(self, hsi):
        x3d = self.input_project(hsi)
        x3d = self.residual(x3d)
        return _group_mean_pool_depth(x3d, self.spec_tokens)


class SpectralTokenPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.score = nn.Conv3d(1, 1, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=True)

    def forward(self, x3d):
        scores = self.score(x3d.mean(dim=1, keepdim=True))
        weights = torch.softmax(scores, dim=2)
        return (x3d * weights).sum(dim=2)


class PCASpatialStem(nn.Module):
    def __init__(self, in_channels, embed_dim, stem_dim=None):
        super().__init__()
        out_dim = stem_dim or embed_dim
        self.input_project = nn.Sequential(
            nn.Conv2d(in_channels, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )
        self.residual = DepthwiseResidual2d(out_dim)

    def forward(self, hsi_pca):
        return self.residual(self.input_project(hsi_pca))


class AuxiliarySpatialStem(nn.Module):
    def __init__(self, in_channels, embed_dim, stem_dim=None):
        super().__init__()
        out_dim = stem_dim or embed_dim
        self.input_project = nn.Sequential(
            nn.Conv2d(in_channels, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )
        self.residual = DepthwiseResidual2d(out_dim)

    def forward(self, aux):
        return self.residual(self.input_project(aux))


class MultiSource3DFusion(nn.Module):
    def __init__(self, hsi_dim, side_dim, embed_dim):
        super().__init__()
        attn_channels = 48
        self.pca_to_hsi = CovCrossAttention3d(query_channels=hsi_dim, context_channels=side_dim)
        self.aux_to_hsi = CovCrossAttention3d(query_channels=hsi_dim, context_channels=side_dim)
        self.mid_project = nn.Sequential(
            nn.Conv3d(hsi_dim, attn_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(attn_channels),
            nn.GELU(),
        )
        self.pca_redistribute = nn.Conv3d(attn_channels, side_dim, kernel_size=1, bias=False)
        self.aux_redistribute = nn.Conv3d(attn_channels, side_dim, kernel_size=1, bias=False)
        self.input_project = nn.Sequential(
            nn.Conv3d(side_dim * 2, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm3d(embed_dim),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, bias=False),
            nn.BatchNorm3d(embed_dim),
            nn.GELU(),
        )

    def _redistribute(self, attention, mid_feature, out_project):
        b, _, depth, height, width = mid_feature.shape
        mid_flat = mid_feature.flatten(2)
        enhanced = torch.matmul(attention.transpose(1, 2), mid_flat).view(b, -1, depth, height, width)
        return out_project(enhanced)

    def forward(self, hsi_3d, pca_spatial, aux_spatial):
        depth = hsi_3d.shape[2]
        pca_3d = pca_spatial.unsqueeze(2).expand(-1, -1, depth, -1, -1)
        aux_3d = aux_spatial.unsqueeze(2).expand(-1, -1, depth, -1, -1)
        pca_hsi, pca_attention = self.pca_to_hsi(hsi_3d, pca_3d, return_attention=True)
        aux_hsi, aux_attention = self.aux_to_hsi(hsi_3d, aux_3d, return_attention=True)
        mid_feature = self.mid_project(hsi_3d + pca_hsi + aux_hsi)
        pca_enhanced = pca_3d + self._redistribute(pca_attention, mid_feature, self.pca_redistribute)
        aux_enhanced = aux_3d + self._redistribute(aux_attention, mid_feature, self.aux_redistribute)
        fused = torch.cat([pca_enhanced, aux_enhanced], dim=1)
        fused = self.input_project(fused)
        return self.fuse(fused)


class DualSource2DFusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.pca_to_aux = CovCrossAttention2d(query_channels=embed_dim, context_channels=embed_dim)
        self.aux_to_pca = CovCrossAttention2d(query_channels=embed_dim, context_channels=embed_dim)
        self.input_project = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.fuse = DepthwiseResidual2d(embed_dim)

    def forward(self, pca_spatial, aux_spatial):
        pca_enhanced = pca_spatial + self.aux_to_pca(pca_spatial, aux_spatial)
        aux_enhanced = aux_spatial + self.pca_to_aux(aux_spatial, pca_spatial)
        fused = torch.cat([pca_enhanced, aux_enhanced], dim=1)
        return self.fuse(self.input_project(fused))


class HSIInitialEnhancement(nn.Module):
    def __init__(self, hsi_dim, embed_dim):
        super().__init__()
        self.pca_from_hsi = CovCrossAttention2d(query_channels=embed_dim, context_channels=hsi_dim)
        self.aux_from_hsi = CovCrossAttention2d(query_channels=embed_dim, context_channels=hsi_dim)

    def forward(self, hsi_context, pca_spatial, aux_spatial):
        pca_spatial = pca_spatial + self.pca_from_hsi(pca_spatial, hsi_context)
        aux_spatial = aux_spatial + self.aux_from_hsi(aux_spatial, hsi_context)
        return pca_spatial, aux_spatial


class PCAXStage(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.pca_to_aux = CovCrossAttention2d(query_channels=embed_dim, context_channels=embed_dim)
        self.aux_to_pca = CovCrossAttention2d(query_channels=embed_dim, context_channels=embed_dim)
        self.pca_model = PCAHilbertVolumeBlock(embed_dim)
        self.aux_model = AuxSpatialMamba2DBlock(embed_dim)

    def forward(self, pca_spatial, aux_spatial):
        pca_enhanced = pca_spatial + self.aux_to_pca(pca_spatial, aux_spatial)
        aux_enhanced = aux_spatial + self.pca_to_aux(aux_spatial, pca_spatial)
        return self.pca_model(pca_enhanced), self.aux_model(aux_enhanced)


class DualBranchFusionHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.pca_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
        )
        self.aux_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, num_classes),
        )

    def forward(self, pca_global, aux_global):
        pca_global = self.pca_proj(pca_global)
        aux_global = self.aux_proj(aux_global)
        fused = torch.cat([pca_global, aux_global, pca_global * aux_global], dim=-1)
        return self.classifier(fused)


class SSFuseMamba(nn.Module):
    def __init__(
        self,
        hsi_channels,
        pca_channels,
        aux_channels,
        num_classes,
        embed_dim=48,
        stem_dim=24,
        spec_tokens=16,
        stage_depths=(2, 2, 2),
    ):
        super().__init__()
        hsi_feature_dim = embed_dim
        stage_count = max(1, len(stage_depths))
        self.hsi_stem = HSI3DStem(embed_dim=embed_dim, stem_dim=stem_dim, spec_tokens=spec_tokens)
        self.hsi_pool = SpectralTokenPool()
        self.pca_stem = PCASpatialStem(in_channels=pca_channels, embed_dim=embed_dim, stem_dim=stem_dim)
        self.x_stem = AuxiliarySpatialStem(in_channels=aux_channels, embed_dim=embed_dim, stem_dim=stem_dim)
        self.hsi_align = nn.Sequential(
            nn.Conv2d(stem_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.pca_align = nn.Sequential(
            nn.Conv2d(stem_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.aux_align = nn.Sequential(
            nn.Conv2d(stem_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.input_enhancement = HSIInitialEnhancement(hsi_dim=hsi_feature_dim, embed_dim=embed_dim)
        self.stages = nn.ModuleList([PCAXStage(embed_dim) for _ in range(stage_count)])
        self.pca_token_pool = AuxSpatialMeanPooling(embed_dim=embed_dim)
        self.aux_token_pool = AuxSpatialMeanPooling(embed_dim=embed_dim)
        self.head = DualBranchFusionHead(embed_dim=embed_dim, num_classes=num_classes)
        _ = hsi_channels

    def forward(self, hsi, hsi_pca, aux):
        pca_spatial = self.pca_align(self.pca_stem(hsi_pca))
        aux_spatial = self.aux_align(self.x_stem(aux))
        # Input HSI-guided enhancement is kept in the module for ablation, but bypassed here.
        _ = hsi
        for stage in self.stages:
            pca_spatial, aux_spatial = stage(pca_spatial, aux_spatial)
        pca_global = self.pca_token_pool(pca_spatial)
        aux_global = self.aux_token_pool(aux_spatial)
        return self.head(pca_global, aux_global)
