import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


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


class SimpleEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


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


class SpectralSpatialFourierEnhancement(nn.Module):
    def __init__(self, cutoff=0.45):
        super().__init__()
        self.cutoff = cutoff
        self.low_weight = nn.Parameter(torch.zeros(1))
        self.high_weight = nn.Parameter(torch.zeros(1))

    def _frequency_masks(self, depth, height, width_freq, device, dtype):
        spectral_axis = torch.fft.fftfreq(depth, device=device).abs().to(dtype).view(depth, 1, 1)
        y_axis = torch.fft.fftfreq(height, device=device).abs().to(dtype).view(1, height, 1)
        x_axis = torch.fft.rfftfreq((width_freq - 1) * 2, device=device).to(dtype).view(1, 1, width_freq)
        max_radius = torch.sqrt(torch.tensor(0.75, device=device, dtype=dtype))
        radius = torch.sqrt(spectral_axis * spectral_axis + y_axis * y_axis + x_axis * x_axis) / max_radius
        low_mask = (radius <= self.cutoff).to(dtype)
        high_mask = 1.0 - low_mask
        return low_mask.view(1, 1, depth, height, width_freq), high_mask.view(1, 1, depth, height, width_freq)

    def forward(self, hsi):
        depth, height, width = hsi.shape[-3:]
        hsi_freq = torch.fft.rfftn(hsi, dim=(-3, -2, -1), norm="ortho")
        low_mask, high_mask = self._frequency_masks(
            depth,
            height,
            hsi_freq.shape[-1],
            hsi.device,
            hsi_freq.real.dtype,
        )
        low = torch.fft.irfftn(hsi_freq * low_mask, s=(depth, height, width), dim=(-3, -2, -1), norm="ortho")
        high = torch.fft.irfftn(hsi_freq * high_mask, s=(depth, height, width), dim=(-3, -2, -1), norm="ortho")
        low_weight = torch.sigmoid(self.low_weight).view(1, 1, 1, 1, 1)
        high_weight = torch.sigmoid(self.high_weight).view(1, 1, 1, 1, 1)
        return hsi + low_weight * low + high_weight * high


class MambaTokenMixer(nn.Module):
    def __init__(self, embed_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        if Mamba is not None:
            self.mixer = Mamba(d_model=embed_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.mixer = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Linear(embed_dim * 2, embed_dim),
            )

    def forward(self, tokens):
        return tokens + self.mixer(self.norm(tokens))


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
        sequence = F.adaptive_avg_pool1d(sequence.transpose(1, 2), self.out_tokens)
        return sequence.transpose(1, 2)


class CovarianceChannelAttention3d(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.out_proj = nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, depth, height, width = x.shape
        n = depth * height * width
        flat = x.flatten(2)
        normed = self.norm(flat.transpose(1, 2)).transpose(1, 2)
        centered = normed - normed.mean(dim=-1, keepdim=True)
        covariance = torch.matmul(centered, centered.transpose(1, 2)) / max(n - 1, 1)
        attention = torch.softmax(covariance, dim=-1)
        mixed = torch.matmul(attention, flat).view(b, c, depth, height, width)
        return x + self.gamma.view(1, 1, 1, 1, 1) * self.out_proj(mixed)


class HilbertLocalContext3d(nn.Module):
    def __init__(self, embed_dim, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.local_mix = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size, padding=padding, groups=embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(embed_dim),
        )
        self.gamma = nn.Parameter(torch.zeros(1))
        self.act = nn.GELU()
        self._index_cache = {}

    def _scan_index(self, depth, height, width, device):
        key = (depth, height, width)
        index = self._index_cache.get(key)
        if index is None:
            index = torch.tensor(_hilbert_scan_indices_3d(depth, height, width), dtype=torch.long)
            self._index_cache[key] = index
        return index.to(device=device)

    def forward(self, x):
        b, c, depth, height, width = x.shape
        scan_index = self._scan_index(depth, height, width, x.device)
        flat = x.flatten(2)
        sequence = flat.index_select(dim=2, index=scan_index)
        mixed_sequence = self.local_mix(sequence)
        restored = flat.new_empty(flat.shape)
        scatter_index = scan_index.view(1, 1, -1).expand(b, c, -1)
        restored.scatter_(dim=2, index=scatter_index, src=mixed_sequence)
        restored = restored.view(b, c, depth, height, width)
        return self.act(x + self.gamma.view(1, 1, 1, 1, 1) * restored)


class HSI3DStem(nn.Module):
    def __init__(self, embed_dim, stem_dim=24, spec_tokens=16):
        super().__init__()
        self.spec_tokens = spec_tokens
        self.spectral_spatial = ConvBNGELU3d(1, stem_dim, kernel_size=(7, 3, 3), padding=(3, 1, 1))
        self.depthwise_mix = ConvBNGELU3d(stem_dim, stem_dim, kernel_size=(3, 3, 3), padding=1, groups=stem_dim)
        self.pointwise_expand = ConvBNGELU3d(stem_dim, stem_dim * 2, kernel_size=1)
        self.spectral_down = nn.Sequential(
            ConvBNGELU3d(stem_dim * 2, stem_dim * 2, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0), groups=stem_dim * 2),
            ConvBNGELU3d(stem_dim * 2, stem_dim * 2, kernel_size=1),
            ConvBNGELU3d(stem_dim * 2, stem_dim * 2, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0), groups=stem_dim * 2),
        )
        _ = embed_dim

    def forward(self, hsi):
        x3d = self.spectral_spatial(hsi)
        x3d = self.depthwise_mix(x3d)
        x3d = self.pointwise_expand(x3d)
        x3d = self.spectral_down(x3d)
        return F.adaptive_avg_pool3d(x3d, (self.spec_tokens, x3d.shape[-2], x3d.shape[-1]))


class PCASpatialStem(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, hsi_pca):
        return self.encoder(hsi_pca)


class AuxiliarySpatialStem(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.encoder = SimpleEncoder(in_channels, embed_dim)

    def forward(self, aux):
        return self.encoder(aux)


class MultiSource3DFusion(nn.Module):
    def __init__(self, hsi_dim, side_dim, embed_dim):
        super().__init__()
        fused_channels = hsi_dim + side_dim * 2
        self.source_logits = nn.Parameter(torch.log(torch.tensor([0.4, 0.3, 0.3], dtype=torch.float32)))
        self.covariance_attention = CovarianceChannelAttention3d(fused_channels)
        self.input_project = nn.Sequential(
            nn.Conv3d(fused_channels, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm3d(embed_dim),
            nn.GELU(),
        )
        self.hilbert_local = HilbertLocalContext3d(embed_dim)
        self.fuse = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, bias=False),
            nn.BatchNorm3d(embed_dim),
            nn.GELU(),
        )

    def forward(self, hsi_3d, pca_spatial, aux_spatial):
        depth = hsi_3d.shape[2]
        pca_3d = pca_spatial.unsqueeze(2).expand(-1, -1, depth, -1, -1)
        aux_3d = aux_spatial.unsqueeze(2).expand(-1, -1, depth, -1, -1)
        weights = torch.softmax(self.source_logits, dim=0)
        weighted_sources = [
            hsi_3d * weights[0].view(1, 1, 1, 1, 1),
            pca_3d * weights[1].view(1, 1, 1, 1, 1),
            aux_3d * weights[2].view(1, 1, 1, 1, 1),
        ]
        fused = torch.cat(weighted_sources, dim=1)
        fused = self.covariance_attention(fused)
        fused = self.input_project(fused)
        fused = self.hilbert_local(fused)
        return self.fuse(fused)


class SSFuseFusionHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.spatial_proj = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.GELU())
        self.token_proj = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.GELU())
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, num_classes),
        )

    def forward(self, fusion_spatial, spec_tokens):
        spatial_global = self.spatial_proj(fusion_spatial.mean(dim=(-1, -2)))
        token_global = self.token_proj(spec_tokens.mean(dim=1))
        fused = torch.cat([spatial_global, token_global, spatial_global * token_global], dim=-1)
        return self.classifier(fused)


class SSFuseMamba(nn.Module):
    def __init__(
        self,
        hsi_channels,
        pca_channels,
        aux_channels,
        num_classes,
        embed_dim=96,
        stem_dim=24,
        spec_tokens=16,
        stage_depths=(2, 2, 2),
    ):
        super().__init__()
        _ = hsi_channels
        hsi_feature_dim = stem_dim * 2
        self.hsi_stem = HSI3DStem(embed_dim=embed_dim, stem_dim=stem_dim, spec_tokens=spec_tokens)
        self.pca_stem = PCASpatialStem(in_channels=pca_channels, embed_dim=embed_dim)
        self.x_stem = AuxiliarySpatialStem(in_channels=aux_channels, embed_dim=embed_dim)
        self.source_fusion = MultiSource3DFusion(
            hsi_dim=hsi_feature_dim,
            side_dim=embed_dim,
            embed_dim=embed_dim,
        )
        self.hilbert_mamba = HilbertMamba3DModeling(embed_dim=embed_dim, out_tokens=spec_tokens)
        self.spatial_refine = DepthwiseResidual2d(embed_dim)
        self.head = SSFuseFusionHead(embed_dim=embed_dim, num_classes=num_classes)
        _ = stage_depths

    def forward(self, hsi, hsi_pca, aux):
        hsi_3d = self.hsi_stem(hsi)
        pca_spatial = self.pca_stem(hsi_pca)
        aux_spatial = self.x_stem(aux)
        fused_3d = self.source_fusion(hsi_3d, pca_spatial, aux_spatial)
        spec_tokens = self.hilbert_mamba(fused_3d)
        fusion_spatial = self.spatial_refine(fused_3d.mean(dim=2))
        return self.head(fusion_spatial, spec_tokens)
