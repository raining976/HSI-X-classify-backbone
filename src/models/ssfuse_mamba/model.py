import torch
import torch.nn as nn
from mamba_ssm import Mamba


def _center_spiral_scan_indices_2d(height, width):
    y, x = height // 2, width // 2
    order = []
    visited = set()

    def add_if_valid(row, col):
        if 0 <= row < height and 0 <= col < width and (row, col) not in visited:
            order.append(row * width + col)
            visited.add((row, col))

    add_if_valid(y, x)
    directions = ((0, 1), (1, 0), (0, -1), (-1, 0))
    step = 1
    direction_idx = 0
    while len(order) < height * width:
        for _ in range(2):
            dy, dx = directions[direction_idx % 4]
            for _ in range(step):
                y += dy
                x += dx
                add_if_valid(y, x)
                if len(order) == height * width:
                    break
            direction_idx += 1
            if len(order) == height * width:
                break
        step += 1
    return order


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


class CenterSpiralBidirectionalScan2d(nn.Module):
    def __init__(self, embed_dim, token_output_project=False):
        super().__init__()
        self.input_project = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.sequence_mixer = MambaTokenMixer(embed_dim)
        self.token_output_project = nn.Linear(embed_dim, embed_dim) if token_output_project else nn.Identity()
        self.forward_alpha_logit = nn.Parameter(torch.zeros(1))
        self._index_cache = {}

    def _scan_indices(self, height, width, device):
        key = (height, width)
        indices = self._index_cache.get(key)
        if indices is None:
            spiral = _center_spiral_scan_indices_2d(height, width)
            indices = (
                torch.tensor(spiral, dtype=torch.long),
                torch.tensor(list(reversed(spiral)), dtype=torch.long),
            )
            self._index_cache[key] = indices
        return tuple(index.to(device=device) for index in indices)

    def _scan(self, x_flat, scan_index, height, width):
        b, c, _ = x_flat.shape
        sequence = x_flat.index_select(dim=2, index=scan_index).transpose(1, 2)
        sequence = self.sequence_mixer(self.input_project(sequence))
        sequence = self.token_output_project(sequence)
        restored = x_flat.new_empty(b, height * width, c)
        restored.scatter_(dim=1, index=scan_index.view(1, -1, 1).expand(b, -1, c), src=sequence)
        return restored.transpose(1, 2).view(b, c, height, width)

    def forward(self, x):
        _, _, height, width = x.shape
        x_flat = x.flatten(2)
        spiral, spiral_reverse = self._scan_indices(height, width, x.device)
        alpha = torch.sigmoid(self.forward_alpha_logit)
        forward_feature = self._scan(x_flat, spiral, height, width)
        reverse_feature = self._scan(x_flat, spiral_reverse, height, width)
        return alpha * forward_feature + (1.0 - alpha) * reverse_feature


class PCAHilbertVolumeBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.scanner = CenterSpiralBidirectionalScan2d(embed_dim, token_output_project=True)
        self._index_cache = self.scanner._index_cache

    @property
    def output_project(self):
        return self.scanner.token_output_project

    def forward(self, x2d):
        return x2d + self.scanner(x2d)


class AuxSpatialMamba2DBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.scanner = CenterSpiralBidirectionalScan2d(embed_dim)
        self.output_project = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
        )
        self.act = nn.GELU()
        self._index_cache = self.scanner._index_cache

    @property
    def hilbert_mixer(self):
        return self.scanner.sequence_mixer

    def forward(self, x):
        feature = self.scanner(x)
        return self.act(x + self.output_project(feature))


class AuxSpatialMeanPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        _ = embed_dim

    def forward(self, x):
        return x.mean(dim=(2, 3))


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


class PCAXStage(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.pca_to_aux = CovCrossAttention2d(query_channels=embed_dim, context_channels=embed_dim)
        self.aux_to_pca = CovCrossAttention2d(query_channels=embed_dim, context_channels=embed_dim)
        self.aux_to_pca_scale = nn.Parameter(torch.tensor(0.1))
        self.pca_to_aux_scale = nn.Parameter(torch.tensor(0.1))
        self.pca_model = PCAHilbertVolumeBlock(embed_dim)
        self.aux_model = AuxSpatialMamba2DBlock(embed_dim)

    def forward(self, pca_spatial, aux_spatial):
        pca_enhanced = pca_spatial + self.aux_to_pca_scale * self.aux_to_pca(pca_spatial, aux_spatial)
        aux_enhanced = aux_spatial + self.pca_to_aux_scale * self.pca_to_aux(aux_spatial, pca_spatial)
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
        pca_channels,
        aux_channels,
        num_classes,
        embed_dim=48,
        stem_dim=24,
        stage_depths=(1),
    ):
        super().__init__()
        stage_count = max(1, len(stage_depths))
        self.pca_stem = PCASpatialStem(in_channels=pca_channels, embed_dim=embed_dim, stem_dim=stem_dim)
        self.x_stem = AuxiliarySpatialStem(in_channels=aux_channels, embed_dim=embed_dim, stem_dim=stem_dim)
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
        self.stages = nn.ModuleList([PCAXStage(embed_dim) for _ in range(stage_count)])
        self.pca_token_pool = AuxSpatialMeanPooling(embed_dim=embed_dim)
        self.aux_token_pool = AuxSpatialMeanPooling(embed_dim=embed_dim)
        self.head = DualBranchFusionHead(embed_dim=embed_dim, num_classes=num_classes)

    def forward(self, hsi_pca, aux):
        pca_spatial = self.pca_align(self.pca_stem(hsi_pca))
        aux_spatial = self.aux_align(self.x_stem(aux))
        for stage in self.stages:
            pca_spatial, aux_spatial = stage(pca_spatial, aux_spatial)
        pca_global = self.pca_token_pool(pca_spatial)
        aux_global = self.aux_token_pool(aux_spatial)
        return self.head(pca_global, aux_global)
