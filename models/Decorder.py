import math
import warnings

import torch
import torch.nn as nn

from timm.models.layers import DropPath, trunc_normal_
import numpy as np


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.v_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, v):
        B, N, _ = q.shape
        C = self.out_dim
        k = v
        NK = k.size(1)

        q = self.q_map(q).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_map(k).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_map(v).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, dim_q=None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        # self.query_p = nn.Embedding(1,768)

        dim_q = dim_q or dim
        self.norm_q = norm_layer(dim_q)
        self.norm_v = norm_layer(dim_q)
        self.norm_att = norm_layer(dim_q)
        self.norm_catt = norm_layer(dim_q)
        self.norm_mlp = norm_layer(dim_q)
        self.att = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.croatt = CrossAttention(
            dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, q, v):
        # query_p = self.query_p.weight
        # B,N,_ = v.shape
        # query_p = query_p.unsqueeze(0).repeat(v.shape[0],N,1)

        # q = self.norm_att(q)
        # q = q + query_p
        # q = q+ self.drop_path(self.att(q))
        # q = q + self.norm_catt(q)
        # q = q + query_p
        # q = q + self.drop_path(self.croatt(q, v + pos_v))
        # q = q + self.norm_mlp(q)
        # q = q + query_p
        # q = q + self.drop_path(self.mlp(q))
        q = q + self.drop_path(self.croatt(self.norm_q(q), self.norm_v(v)))
        q = q + self.drop_path(self.mlp(self.norm2(q)))
        return q


class decoder_fuser(nn.Module):
    def __init__(self, in_dim, dim, num_heads, num_layers):
        super(decoder_fuser, self).__init__()
        model_list = []
        for i in range(num_layers):
            model_list.append(DecoderBlock(dim, num_heads))
        self.model = nn.ModuleList(model_list)
        self.pos = nn.Parameter(torch.randn(num_heads, in_dim//num_heads))
        self.querys = nn.Parameter(torch.randn(num_heads*2, in_dim//num_heads))
        # querys
        # self.querys = nn.Parameter(torch.zeros(1, 5, dim))

        # 正态分布取值
        # def _no_grad_trunc_normal_(tensor, mean, std, a, b):
        #     def norm_cdf(x):
        #         # Computes standard normal cumulative distribution function
        #         return (1. + math.erf(x / math.sqrt(2.))) / 2.
        #
        #     if (mean < a - 2 * std) or (mean > b + 2 * std):
        #         warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
        #                       "The distribution of values may be incorrect.",
        #                       stacklevel=2)
        #
        #     with torch.no_grad():
        #         # Values are generated by using a truncated uniform distribution and
        #         # then using the inverse CDF for the normal distribution.
        #         # Get upper and lower cdf values
        #         l = norm_cdf((a - mean) / std)
        #         u = norm_cdf((b - mean) / std)
        #
        #         # Uniformly fill tensor with values from [l, u], then translate to
        #         # [2l-1, 2u-1].
        #         tensor.uniform_(2 * l - 1, 2 * u - 1)
        #
        #         # Use inverse cdf transform for normal distribution to get truncated
        #         # standard normal
        #         tensor.erfinv_()
        #
        #         # Transform to proper mean, std
        #         tensor.mul_(std * math.sqrt(2.))
        #         tensor.add_(mean)
        #
        #         # Clamp to ensure it's in the proper range
        #         tensor.clamp_(min=a, max=b)
        #         return tensor
        #
        # def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
        #     # type: (Tensor, float, float, float, float) -> Tensor
        #     return _no_grad_trunc_normal_(tensor, mean, std, a, b)
        #
        # trunc_normal_(self.querys, std=.02)

    def forward(self, v):
        bs, c, hw = v.shape
        v = torch.reshape(v, (bs, 8, -1))
        # querys = self.querys.weight
        # querys = querys.unsqueeze(0).repeat(B, 1, 1)
        # q = torch.zeros_like(querys)
        # v = v.permute(2, 0, 1)
        pos = self.pos.expand(bs, -1, -1)
        q = self.querys.expand(bs, -1, -1)
        tgt = v + pos
        for _layer in self.model:
            q = _layer(q, tgt)
        q = q.view(bs, 2, -1)
        return q