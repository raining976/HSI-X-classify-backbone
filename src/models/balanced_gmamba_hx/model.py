import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


class ConvBNGELU2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


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


class LearnableChannelFusion(nn.Module):
    def __init__(self, channels, token_mode=False):
        super().__init__()
        self.token_mode = token_mode
        self.alpha = nn.Parameter(torch.zeros(channels))

    def forward(self, a, b):
        if self.token_mode:
            weight = torch.sigmoid(self.alpha).view(1, 1, -1)
        else:
            weight = torch.sigmoid(self.alpha).view(1, -1, 1, 1)
        return weight * a + (1.0 - weight) * b


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


class HSIBalancedStem(nn.Module):
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
        self.spatial_project = nn.Sequential(
            nn.Conv2d(stem_dim * 2 * spec_tokens, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.token_project = nn.Sequential(
            nn.Linear(stem_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, hsi):
        x3d = self.spectral_spatial(hsi)
        x3d = self.depthwise_mix(x3d)
        x3d = self.pointwise_expand(x3d)
        x3d = self.spectral_down(x3d)
        x3d = F.adaptive_avg_pool3d(x3d, (self.spec_tokens, x3d.shape[-2], x3d.shape[-1]))
        b, c, s, h, w = x3d.shape
        spatial = self.spatial_project(x3d.reshape(b, c * s, h, w))
        tokens = self.token_project(x3d.mean(dim=(-1, -2)).transpose(1, 2))
        return spatial, tokens


class PCABalancedTokenStem(nn.Module):
    def __init__(self, in_channels, embed_dim, token_count=16):
        super().__init__()
        self.token_count = token_count
        self.encoder = nn.Sequential(
            ConvBNGELU2d(in_channels, embed_dim, kernel_size=3, padding=1),
            ConvBNGELU2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim),
        )
        self.token_project = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, hsi_pca):
        features = self.encoder(hsi_pca)
        pooled = F.adaptive_avg_pool2d(features, (self.token_count, 1)).squeeze(-1).transpose(1, 2)
        return self.token_project(pooled)


class XBalancedStem(nn.Module):
    def __init__(self, in_channels, embed_dim, token_count=16):
        super().__init__()
        self.token_count = token_count
        self.encoder = SimpleEncoder(in_channels, embed_dim)
        self.token_project = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, aux):
        spatial = self.encoder(aux)
        pooled = F.adaptive_avg_pool2d(spatial, (self.token_count, 1)).squeeze(-1).transpose(1, 2)
        tokens = self.token_project(pooled)
        return spatial, tokens


class HSISpectralGuide(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.low_proj = nn.Linear(embed_dim, embed_dim)
        self.high_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )

    def forward(self, tokens):
        x = tokens.transpose(1, 2)
        x_freq = torch.fft.rfft(x, dim=-1, norm="ortho")
        length = x_freq.shape[-1]
        split = max(1, length // 2)

        low_mask = torch.zeros_like(x_freq.real)
        high_mask = torch.zeros_like(x_freq.real)
        low_mask[..., :split] = 1.0
        high_mask[..., split:] = 1.0
        if high_mask.sum() == 0:
            high_mask[..., -1:] = 1.0
            low_mask[..., -1:] = 0.0

        low = torch.fft.irfft(x_freq * low_mask, n=x.shape[-1], dim=-1, norm="ortho").transpose(1, 2)
        high = torch.fft.irfft(x_freq * high_mask, n=x.shape[-1], dim=-1, norm="ortho").transpose(1, 2)
        low = self.out_proj(self.low_proj(low))
        high = self.out_proj(self.high_proj(high))
        low_guide = torch.sigmoid(low.mean(dim=1)).unsqueeze(-1).unsqueeze(-1)
        high_guide = torch.sigmoid(high.mean(dim=1)).unsqueeze(-1).unsqueeze(-1)
        return low, high, low_guide, high_guide


class DualModalSpatialFrequencyDecomposer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.hsi_low_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False)
        self.hsi_high_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False)
        self.x_low_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False)
        self.x_high_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False)

    def _split(self, x, low_proj, high_proj):
        _, _, h, w = x.shape
        x_freq = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        y = torch.linspace(0.0, 1.0, steps=h, device=x.device, dtype=x_freq.real.dtype).view(h, 1)
        x_axis = torch.linspace(0.0, 1.0, steps=x_freq.shape[-1], device=x.device, dtype=x_freq.real.dtype).view(1, -1)
        radial = torch.sqrt(y * y + x_axis * x_axis)
        low_mask = (radial <= 0.45).to(x_freq.real.dtype)
        high_mask = (radial > 0.45).to(x_freq.real.dtype)
        low = torch.fft.irfft2(x_freq * low_mask.view(1, 1, h, -1), s=(h, w), dim=(-2, -1), norm="ortho")
        high = torch.fft.irfft2(x_freq * high_mask.view(1, 1, h, -1), s=(h, w), dim=(-2, -1), norm="ortho")
        return low_proj(low), high_proj(high)

    def forward(self, hsi_spatial, x_spatial):
        hsi_low, hsi_high = self._split(hsi_spatial, self.hsi_low_proj, self.hsi_high_proj)
        x_low, x_high = self._split(x_spatial, self.x_low_proj, self.x_high_proj)
        return hsi_low, hsi_high, x_low, x_high


class SpatialMamba2D(nn.Module):
    def __init__(self, embed_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.row_norm = nn.LayerNorm(embed_dim)
        self.col_norm = nn.LayerNorm(embed_dim)
        if Mamba is not None:
            self.row_mixer = Mamba(d_model=embed_dim, d_state=d_state, d_conv=d_conv, expand=expand)
            self.col_mixer = Mamba(d_model=embed_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.row_mixer = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Linear(embed_dim * 2, embed_dim),
            )
            self.col_mixer = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Linear(embed_dim * 2, embed_dim),
            )
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        row_tokens = x.permute(0, 2, 3, 1).reshape(b * h, w, c)
        row_fwd = row_tokens + self.row_mixer(self.row_norm(row_tokens))
        row_fwd = row_fwd.reshape(b, h, w, c).permute(0, 3, 1, 2)

        row_rev_tokens = torch.flip(row_tokens, dims=[1])
        row_rev = row_rev_tokens + self.row_mixer(self.row_norm(row_rev_tokens))
        row_rev = torch.flip(row_rev, dims=[1]).reshape(b, h, w, c).permute(0, 3, 1, 2)

        col_tokens = x.permute(0, 3, 2, 1).reshape(b * w, h, c)
        col_fwd = col_tokens + self.col_mixer(self.col_norm(col_tokens))
        col_fwd = col_fwd.reshape(b, w, h, c).permute(0, 3, 2, 1)

        col_rev_tokens = torch.flip(col_tokens, dims=[1])
        col_rev = col_rev_tokens + self.col_mixer(self.col_norm(col_rev_tokens))
        col_rev = torch.flip(col_rev, dims=[1]).reshape(b, w, h, c).permute(0, 3, 2, 1)
        return self.fuse(torch.cat([row_fwd, row_rev, col_fwd, col_rev], dim=1))


class BalancedGMambaBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.hsi_local = DepthwiseResidual2d(embed_dim)
        self.x_local = DepthwiseResidual2d(embed_dim)
        self.hsi_pca_fusion = LearnableChannelFusion(embed_dim, token_mode=True)
        self.hsi_spectral_guide = HSISpectralGuide(embed_dim)
        self.spatial_decomposer = DualModalSpatialFrequencyDecomposer(embed_dim)
        self.hsi_band_fusion = LearnableChannelFusion(embed_dim)
        self.x_band_fusion = LearnableChannelFusion(embed_dim)
        self.hsi_x_spatial_fusion = LearnableChannelFusion(embed_dim)
        self.x_low_weight = nn.Parameter(torch.zeros(embed_dim))
        self.x_high_weight = nn.Parameter(torch.zeros(embed_dim))
        self.spatial_scan = SpatialMamba2D(embed_dim)
        self.token_mixer = MambaTokenMixer(embed_dim)
        self.out_act = nn.GELU()

    def forward(self, hsi_spatial, x_spatial, spec_tokens, pca_tokens):
        residual_spatial = hsi_spatial
        residual_tokens = spec_tokens
        hsi_spatial = self.hsi_local(hsi_spatial)
        x_spatial = self.x_local(x_spatial)

        spec_tokens = self.hsi_pca_fusion(spec_tokens, pca_tokens)
        hsi_spec_low, hsi_spec_high, low_guide, high_guide = self.hsi_spectral_guide(spec_tokens)
        hsi_low, hsi_high, x_low, x_high = self.spatial_decomposer(hsi_spatial, x_spatial)

        hsi_freq = self.hsi_band_fusion(hsi_low * low_guide, hsi_high * high_guide)
        x_low_weight = torch.sigmoid(self.x_low_weight).view(1, -1, 1, 1)
        x_high_weight = torch.sigmoid(self.x_high_weight).view(1, -1, 1, 1)
        x_freq = self.x_band_fusion(x_low * x_low_weight, x_high * x_high_weight)
        fusion_spatial = self.hsi_x_spatial_fusion(hsi_freq, x_freq)
        fusion_spatial = self.out_act(residual_spatial + fusion_spatial + self.spatial_scan(fusion_spatial))

        spec_tokens = self.token_mixer(spec_tokens + hsi_spec_low + hsi_spec_high)
        spec_tokens = spec_tokens + residual_tokens
        return fusion_spatial, spec_tokens


class BalancedGMambaStage(nn.Module):
    def __init__(self, embed_dim, depth):
        super().__init__()
        self.blocks = nn.ModuleList([BalancedGMambaBlock(embed_dim) for _ in range(depth)])

    def forward(self, hsi_spatial, x_spatial, spec_tokens, pca_tokens):
        for block in self.blocks:
            hsi_spatial, spec_tokens = block(hsi_spatial, x_spatial, spec_tokens, pca_tokens)
        return hsi_spatial, spec_tokens


class DownsampleStage(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.spatial_down = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )
        self.x_down = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )
        self.token_spec = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim))
        self.token_pca = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim))

    def forward(self, fusion_spatial, x_spatial, spec_tokens, pca_tokens):
        return (
            self.spatial_down(fusion_spatial),
            self.x_down(x_spatial),
            self.token_spec(spec_tokens),
            self.token_pca(pca_tokens),
        )


class BalancedFusionHead(nn.Module):
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


class BalancedGMambaHX(nn.Module):
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
        self.hsi_stem = HSIBalancedStem(embed_dim=embed_dim, stem_dim=stem_dim, spec_tokens=spec_tokens)
        self.pca_stem = PCABalancedTokenStem(in_channels=pca_channels, embed_dim=embed_dim, token_count=spec_tokens)
        self.x_stem = XBalancedStem(in_channels=aux_channels, embed_dim=embed_dim, token_count=spec_tokens)
        self.initial_spec_fusion = LearnableChannelFusion(embed_dim, token_mode=True)
        self.stage1 = BalancedGMambaStage(embed_dim=embed_dim, depth=stage_depths[0])
        self.downsample1 = DownsampleStage(embed_dim, embed_dim * 2)
        self.stage2 = BalancedGMambaStage(embed_dim=embed_dim * 2, depth=stage_depths[1])
        self.downsample2 = DownsampleStage(embed_dim * 2, embed_dim * 4)
        self.stage3 = BalancedGMambaStage(embed_dim=embed_dim * 4, depth=stage_depths[2])
        self.head = BalancedFusionHead(embed_dim=embed_dim * 4, num_classes=num_classes)

    def forward(self, hsi, hsi_pca, aux):
        hsi_spatial, hsi_tokens = self.hsi_stem(hsi)
        pca_tokens = self.pca_stem(hsi_pca)
        x_spatial, _ = self.x_stem(aux)
        spec_tokens = self.initial_spec_fusion(hsi_tokens, pca_tokens)

        fusion_spatial, spec_tokens = self.stage1(hsi_spatial, x_spatial, spec_tokens, pca_tokens)
        fusion_spatial, x_spatial, spec_tokens, pca_tokens = self.downsample1(
            fusion_spatial,
            x_spatial,
            spec_tokens,
            pca_tokens,
        )
        fusion_spatial, spec_tokens = self.stage2(fusion_spatial, x_spatial, spec_tokens, pca_tokens)
        fusion_spatial, x_spatial, spec_tokens, pca_tokens = self.downsample2(
            fusion_spatial,
            x_spatial,
            spec_tokens,
            pca_tokens,
        )
        fusion_spatial, spec_tokens = self.stage3(fusion_spatial, x_spatial, spec_tokens, pca_tokens)
        return self.head(fusion_spatial, spec_tokens)
