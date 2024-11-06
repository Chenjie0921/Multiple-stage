
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


class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, dim_q=None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embedding = nn.Embedding(1, dim)
        # self.pos_embedding = nn.Parameter(torch.randn(1,16,2048))
        dim_q = dim_q or dim
        self.norm_att = norm_layer(dim_q)
        self.norm_mlp = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.Crossattn = CrossAttention(
            dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.cls_token = nn.Parameter(torch.randn(1,1,dim))

    def forward(self, v):

        # b,n,_ = v.shape
        # pos = self.pos_embedding.weight
        # pos = pos.unsqueeze(0).repeat(b,n+1,1)
        #
        # cls_token = self.cls_token.expand(b,-1,-1)
        # v = torch.cat((v, cls_token),dim=1)
        # pos = torch.cat((pos_v,pos_cls),dim=1)
        # q = torch.zeros_like(query_p)
        # pos = self.pos_embedding
        # v = self.norm_att(v)
        # # v = v + pos
        # v = v + self.drop_path(self.attn(v))
        # # v = self.norm2(v)
        # v = self.norm_mlp(v)
        # # v = v + pos
        # # q = q + self.drop_path(self.Crossattn(self.norm_q(q), self.norm_v(v)))
        #
        # v = v + self.drop_path(self.mlp(v))

        # enedv = v.mean(dim=1, keepdims = True)# 16 * 1 * 2048
        v = v + self.drop_path(self.attn(self.norm_att(v)))
        v = v + self.drop_path(self.mlp(self.norm2(v)))

        return v


class encoder_fuser(nn.Module):
    def __init__(self, in_dim, dim, num_heads, num_layers):
        super(encoder_fuser, self).__init__()
        model_list = []
        for i in range(num_layers):
            model_list.append(EncoderBlock(dim, num_heads))
        self.model = nn.ModuleList(model_list)

        self.pos_embedding = nn.Parameter(torch.randn(num_heads, in_dim//num_heads))
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
    def forward(self,  v):
        bs, c, hw = v.shape
        v = torch.reshape(v, (bs, 8, -1))
        pos = self.pos_embedding.expand(bs, -1, -1)
        tgt = v + pos
        # pos = self.pos_embedding.weight
        # pos = pos.unsqueeze(0).repeat(b, n+1 , 1)
        # pos_cls = pos.unsqueeze(0).repeat(b,1,1)
        # pos = self.pos_embedding.expand(b, -1, -1)
        # cls_token = self.cls_token.expand(b, -1, -1)
        # v = torch.cat((cls_token, v), dim=1)
        # q = torch.zeros_like(query_p)
        # pos = self.pos_embedding
        # v = self.norm_v(v)
        # v = v + pos
        for _layer in self.model:
            q = _layer(tgt)

        # token = q[:,0,:].unsqueeze(1) #16*2048
        # edfeature = q[:,:-1,:]
        # pos_v = pos[:,0,:].unsqueez e(1)
        q = q.view(bs, 1, -1)
        return q




