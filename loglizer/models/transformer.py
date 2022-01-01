import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.autograd import Variable
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
from math import sqrt

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, in_dim, embed_dim, out_dim, window_size, depth, heads, dim_head, dim_ratio, dropout = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.translayers = nn.ModuleList([])
        self.embed = nn.Linear(in_features= in_dim, out_features= embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.pos_embed = PositionalEncoding(d_model= embed_dim, dropout= dropout)

        for _ in range(depth):
            self.translayers.append(nn.ModuleList([
                PreNorm(embed_dim, Attention(embed_dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(embed_dim, FeedForward(embed_dim, embed_dim * dim_ratio, dropout = dropout))
            ]))

        self.fc1 = nn.Linear(in_features = embed_dim * window_size, out_features= embed_dim * window_size, bias= False)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(in_features = embed_dim * window_size, out_features= out_dim, bias= False)

    def forward(self, x):
        B,L,C = x.shape #[B, window_size, input_size]
        
        x = self.embed(x) #[B, window_size, embed_size]
        x = self.dropout(x)

        x= self.pos_embed(x)

        for attn, ff in self.translayers:
            x = attn(x) + x
            x = ff(x) + x #[B, window_size, embed_size]

        x = x.view(B, self.window_size*self.embed_dim)
        
        x = self.act(self.fc1(x))
        out = self.fc2(x)

        return out