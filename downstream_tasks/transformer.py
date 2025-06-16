import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model) 
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class FrameEncoder(nn.Module):
    """Extract features from a single 64x4x4 frame using CNN."""
    def __init__(self, embed_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128), 
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(1)  # Output: (B, 256, 1, 1)
        )
        self.project = nn.Linear(256, embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x): 
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W) 
        x = self.encoder(x).view(B * T, -1) 
        x = self.project(x) 
        return x.reshape(B, T, -1) 



class ViT(nn.Module):
    def __init__(self, img_size=65536, numpatch=60, in_channels=3,
                 emb_dim=256, depth=2, n_heads=4, mlp_dim=256, n_classes=1, dropout=0.05, max_len=100):
        super().__init__()
      
        self.patch_embed = nn.Linear(512, emb_dim) 
        self.v_embed = nn.Linear(2048, emb_dim )
      #  self.patch_embed = FrameEncoder(embed_dim=emb_dim)  #used if we do the spatial functa
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.numpatch = numpatch
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + numpatch, emb_dim))
        self.dropout = nn.Dropout(dropout)
        self.pos_encoder = PositionalEncoding(emb_dim, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads,
                                                   dim_feedforward=mlp_dim, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, n_classes)

    def forward(self, x, v):
        B = x.shape[0]
        T= x.shape[1]
        x = self.patch_embed(x)  # [B, N, emb_dim]

        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, emb_dim]
        v_token = self.v_embed(v)  #used if we combine phi with v
        x = torch.cat((cls_tokens,  v_token, x), dim=1)  # [B, 1+N, emb_dim]

        x = self.pos_encoder(x)
        x = self.transformer(x)  # [B, T+1, emb_dim]
        x = self.norm(x[:, 0])   # take CLS token output
        logits = self.head(x)
        return logits



