import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=60):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerRegression(nn.Module):
    def __init__(self, input_dim=8192, d_model=1024, nhead=8, num_layers=2, dim_feedforward=512, dropout=0.1):
        super().__init__()
      #  self.input_proj = nn.Sequential(nn.Linear(input_dim, d_model), nn.Tanh(), nn.Linear(d_model, d_model))
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, 60, 8200)
        x = self.input_proj(x)  # (batch_size, 60, d_model)
        x = self.pos_encoder(x)
       
        x = self.transformer_encoder(x)  # (batch_size, 60, d_model)
        

        # Pooling: mean over time
        x = x.mean(dim=1)  # (batch_size, d_model)

        return x, self.regression_head(x)

# Example usaged
model = TransformerRegression()
print(model)

# Dummy input
x = torch.randn(16, 60, 8192)  # batch of 16 sequences
repr, y = model(x)
print("Output shape:", y.shape)  # should be (16, 1)

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=256, n_patches=60, in_channels=1, emb_dim=768):
        super().__init__()
        self.n_patches = n_patches
        self.patch_embed = nn.Linear(256, emb_dim)
    
    def forward(self, x):
        print('x', x.shape)
        x = self.patch_embed(x)  # shape: [B, emb_dim, H', W']
        print('x', x.shape)
        #x = x.flatten(2)         # shape: [B, emb_dim, N]
        x = x.transpose(1, 2)    # shape: [B, N, emb_dim]
        return x

class FrameEncoder(nn.Module):
    """Extract features from a single 64x4x4 frame using CNN."""
    def __init__(self, embed_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128), 
          #  nn.MaxPool2d(2),  # (16x16)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(1)  # Output: (B, 256, 1, 1)
        )
        self.project = nn.Linear(256, embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):  # x: (B, 60, 64, 32, 32)
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W) 
        x = self.encoder(x).view(B * T, -1)  # (B*T, 256)
        x = self.project(x)  # (B*T, embed_dim)
       # x=self.dropout(x)
        return x.reshape(B, T, -1)  # (B, 60, embed_dim)



class TransformerClassifier(nn.Module):
    def __init__(self, num_classes, seq_len=60, embed_dim=256, num_heads=4, num_layers=4):
        super().__init__()
        self.frame_encoder = FrameEncoder(embed_dim=embed_dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cls_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):  # x: (B, 60, 64, 32, 32)
        x = self.frame_encoder(x)  # (B, 60, embed_dim)
        x = x + self.pos_embedding[:, :x.size(1), :]  # add positional encoding
        x = self.transformer(x)  # (B, 60, embed_dim)
        x = x.mean(dim=1)  # global average pooling over time
        return self.cls_head(x)  # (B, num_classes)


class ViT(nn.Module):
    def __init__(self, img_size=65536, numpatch=60, in_channels=3,
                 emb_dim=256, depth=2, n_heads=4, mlp_dim=256, n_classes=1, dropout=0.05, max_len=100):
        super().__init__()
      
        self.patch_embed = nn.Linear(512, emb_dim) 
        self.v_embed = nn.Linear(2048, emb_dim )
      #  self.patch_embed = FrameEncoder(embed_dim=emb_dim)
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
        v_token = self.v_embed(v)
        x = torch.cat((cls_tokens,  v_token, x), dim=1)  # [B, 1+N, emb_dim]

        x = self.pos_encoder(x)
       # x = self.dropout(x)
        x = self.transformer(x)  # [B, T+1, emb_dim]
        x = self.norm(x[:, 0])   # take CLS token output
        logits = self.head(x)
        return logits

# Example usage:
#model = ViT(img_size=224, numpatch=60, in_channels=1, n_classes=1)
#img = torch.rand(8, 60, 512)  # batch of 8 images
#logits = model(img)  # [8, 10]
#print(logits.shape)




class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape [1, max_len, dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, T, D]
        return x + self.pe[:, :x.size(1)]


class Conv3DClassifier(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(32, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),  # (T/2, 16, 16)
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)  # Global pooling
        )
        self.fc = nn.Linear(256, 1)

    def forward(self, x):  # x: (B, T, H, W, C)
        x = x.permute(0, 2, 1, 3, 4)  # (B, 64, 60, 32, 32)

        x = self.conv3d(x).view(x.size(0), -1)
        return self.fc(x)



   