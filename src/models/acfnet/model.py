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


class ACFNet(nn.Module):
    def __init__(self, pca_channels, aux_channels, num_classes, hidden_dim=64):
        super().__init__()
        self.pca_stem = nn.Sequential(
            ConvBlock(pca_channels, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim),
        )
        self.x_stem = nn.Sequential(
            ConvBlock(aux_channels, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim),
        )

        self.pca_enhance = SpatialCrossEnhance(hidden_dim)
        self.x_enhance = ChannelCrossEnhance(hidden_dim)
        self.fusion_mamba = FusionMamba2D(hidden_dim)

        self.pca_model = nn.Sequential(
            ConvBlock(hidden_dim, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim),
        )
        self.x_model = nn.Sequential(
            ConvBlock(hidden_dim, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim),
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
