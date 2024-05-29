
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math

@dataclass
class ViTConfig:
    img_size: int = 256
    patch_size: int = 16
    reg_tokens: int = 0
    n_layers: int = 3
    n_heads: int = 4
    n_embed: int = 64
    in_chans: int = 3
    dropout: float = 0.0
    device: str = 'cpu'
    mlp_ratio: int = 4
    num_classes: int = 10
    bias: bool = True

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super(LayerScale, self).__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init_value)

    def forward(self, x):
        # dot product is equivalent to multiplying by diagonal matrix
        return x * self.gamma

class PatchEmbed(nn.Module):
    def __init__(self, H, W, patch_size=16, in_chans=3, n_embed=100):
        super().__init__()

        self.num_patches = (H * W) // (patch_size ** 2)
        self.patch_size = patch_size

        # conv operation acts as a linear embedding
        self.proj = nn.Conv2d(in_chans, n_embed, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  #(B, C, H, W) -> (B, n_embed, H, W)
        x = x.flatten(2) #(B, n_embed, H, W) -> (B, n_embed, H*W)
        x = x.transpose(1, 2) # (B, n_embed, H*W) -> (B, H*W, n_embed)

        return x

class AcausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.img_size % cfg.n_heads == 0, "img_size must be divisible by n_heads"

        self.n_heads, self.n_embed = cfg.n_heads, cfg.n_embed
        self.head_size = self.n_embed // self.n_heads

        # key,query,value matrices as a single batch for efficiency
        self.qkv = nn.Linear(cfg.n_embed, 3*cfg.n_embed, bias=cfg.bias)

        # output layer
        self.proj = nn.Linear(cfg.n_embed, cfg.n_embed, bias=cfg.bias)

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B,T,C = x.shape

        # get q,k,v, matrices and reshape for multi-head attention
        q, k, v = self.qkv(x).chunk(3, dim=2)
        q, k, v = [z.view(B, T, self.n_heads, self.head_size).transpose(1, 2) for z in (q, k, v)]  # (B, nh, T, hs)
        
        weights = q @ k.transpose(-2,-1) * (1.0 / math.sqrt(k.size(-1))) # (B,T,hs) @ (B,hs,T) --->  (B,T,T)
        weights = F.softmax(weights, dim=-1)
        weights = self.attn_dropout(weights)

        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.proj(out))
        return out 

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        n_embed = cfg.n_embed
        self.attn = AcausalSelfAttention(cfg)
        self.mlp = FeedForward(cfg)

        self.norm1 = nn.LayerNorm(n_embed)
        self.norm2 = nn.LayerNorm(n_embed) 
        self.ls1 = LayerScale(n_embed)
        self.ls2 = LayerScale(n_embed)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n_embed, dropout = cfg.n_embed, cfg.dropout

        self.fc1 = nn.Linear(n_embed, cfg.mlp_ratio * n_embed)
        self.fc2 = nn.Linear(cfg.mlp_ratio * n_embed, n_embed)  # projection layer
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class ViT(nn.Module):
    def __init__(self, cfg):
        super(ViT, self).__init__()

        image_height, image_width = pair(cfg.img_size)
        patch_height, patch_width = pair(cfg.patch_size)
        self.device = cfg.device

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        self.patch_embed = PatchEmbed(cfg.img_size, cfg.img_size, cfg.patch_size, cfg.in_chans, cfg.n_embed)

        self.reg_token = nn.Parameter(torch.zeros(1, cfg.reg_tokens, cfg.n_embed)) 

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, cfg.n_embed) * .02)

        # Sequential blocks for computation 
        self.blocks = nn.Sequential(
            *[Block(cfg) for _ in range(cfg.n_layers)]
        )

        self.fc_norm = nn.LayerNorm(cfg.n_embed)

        # Linear layer to map from token embedding to log_probs
        self.head = nn.Linear(cfg.n_embed, cfg.num_classes)

    def pos_emb_absolute(self, x):
        # Generate position indices (0 to T-1) and retrieve position embeddings
        T = x.shape[1]
        # pos = torch.arange(T, device=self.device).unsqueeze(0)  # (1, T)
        # pos_embed = self.pos_embedding_table(pos)  # (T,C)

        x = x + self.pos_embed

        # add register token
        to_cat = [self.reg_token.expand(x.shape[0], -1, -1)]
        x = torch.cat(to_cat + [x], dim=1)
        return x


    def forward(self, img, targets=None):

        img = img.to(self.device)
        x = self.patch_embed(img)
        # x += self.pos_embedding.type_as(x)
        x = self.pos_emb_absolute(x)

        x = self.blocks(x)
        x = x.mean(dim=1) # global avg pooling
        x = self.fc_norm(x)
        log_probs = self.head(x)

        if targets is not None:
            targets = targets.to(self.device)
            loss = F.cross_entropy(log_probs, targets)
        else:
            loss = None

        return log_probs, loss
