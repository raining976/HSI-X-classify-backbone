import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.res_weight = nn.Parameter(torch.tensor(-2.0))

    def forward(self, pca_feat, x_feat):
        b, c, h, w = pca_feat.shape
        q = self.q_proj(pca_feat).flatten(2).transpose(1, 2)
        k = self.k_proj(x_feat).flatten(2).transpose(1, 2)
        v = self.v_proj(x_feat).flatten(2).transpose(1, 2)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(c), dim=-1)
        enhanced = torch.matmul(attn, v).transpose(1, 2).reshape(b, c, h, w)
        enhanced = self.out_proj(enhanced)
        return pca_feat + torch.sigmoid(self.res_weight) * enhanced


class ChannelCrossEnhance(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.res_weight = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x_feat, pca_feat):
        b, c, h, w = x_feat.shape
        n = h * w
        q = self.q_proj(x_feat).flatten(2)
        k = self.k_proj(pca_feat).flatten(2)
        v = self.v_proj(pca_feat).flatten(2)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(n), dim=-1)
        enhanced = torch.matmul(attn, v).reshape(b, c, h, w)
        enhanced = self.out_proj(enhanced)
        return x_feat + torch.sigmoid(self.res_weight) * enhanced


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

        pca_enhanced = self.pca_enhance(pca_feat, x_feat)
        x_enhanced = self.x_enhance(x_feat, pca_feat)

        pca_out = self.pca_model(pca_enhanced)
        x_out = self.x_model(x_enhanced)
        fused = self.fusion(torch.cat([pca_out, x_out], dim=1))
        return self.classifier(fused)
