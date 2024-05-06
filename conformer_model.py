# -*- coding: utf-8 -*-
'''
@author:xw2941
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward=2048, dropout=0.1, kernel_size=31):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.conv_module = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Conv1d(d_model, 2 * d_model, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(2 * d_model, 2 * d_model, kernel_size=kernel_size, padding=kernel_size//2, groups=d_model),
            nn.GELU(),
            nn.Conv1d(2 * d_model, d_model, kernel_size=1),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.norm1(src)
        src2 = self.mha(src2, src2, src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout(src2)
        src2 = self.conv_module(src.permute(1, 2, 0)).permute(2, 0, 1)
        src = src + self.dropout(src2)
        src2 = self.norm2(src)
        src2 = self.feed_forward(src2)
        src = src + self.dropout(src2)
        return src

class ConformerModel(nn.Module):
    def __init__(self, num_features, d_model, num_blocks, num_classes, n_heads, dim_feedforward=2048, dropout=0.1, kernel_size=31):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, n_heads, dim_feedforward, dropout, kernel_size)
            for _ in range(num_blocks)
        ])
        self.output_proj = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output_proj(x)
        return x
