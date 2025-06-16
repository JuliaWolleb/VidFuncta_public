import math
import torch
from torch import nn
from sklearn.model_selection import KFold

class LatentModulatedSIRENLayer(nn.Module):
    def __init__(self, in_size, out_size, latent_modulation_dim: 512, w0=30.,
                 modulate_shift=True, modulate_scale=False, is_first=False, is_last=False, k=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.latent_modulation_dim = latent_modulation_dim
        self.w0 = w0
        self.modulate_shift = modulate_shift
        self.modulate_scale = modulate_scale
        self.is_last = is_last
        self.k=k

        self.linear = nn.Linear(in_size, out_size)

        if modulate_shift:
            self.modulate_shift_layer = nn.Linear(latent_modulation_dim, out_size)
        if modulate_scale:
            self.modulate_scale_layer = nn.Linear(latent_modulation_dim, out_size)

        self._init(w0, is_first)

    def _init(self, w0, is_first):
        dim_in = self.linear.weight.size(1)
        w_std = 1/dim_in if is_first else (math.sqrt(6.0/dim_in)/w0)
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        nn.init.uniform_(self.linear.bias, -w_std, w_std)

    def hermite(x,self):
        y=torch.exp(-x**2)*torch.sin(2*x)
        return y

    def forward(self, x, latent):
        x = self.linear(x)
        
        if not self.is_last:
            shift = 0.0 if not self.modulate_shift else self.modulate_shift_layer(latent)
            scale = 1.0 if not self.modulate_scale else self.modulate_scale_layer(latent)

            if self.modulate_shift:
                if len(shift.shape) == 2:
                    shift = shift.unsqueeze(dim=1)
            if self.modulate_scale:
                if len(scale.shape) == 2:
                    scale = scale.unsqueeze(dim=1)

            x = scale * x + shift
           
            x = torch.sin(self.w0 * x)
        return x




class LatentModulatedSIRENLayer_v3(nn.Module):
    def __init__(self, in_size, out_size, latent_modulation_dim: 512, latent_v_dim: 512,w0=30.,
                 modulate_shift=True, modulate_scale=False, is_first=False, is_last=False, k=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.latent_modulation_dim = latent_modulation_dim
        self.latent_v_dim = latent_v_dim
        self.w0 = w0
        self.modulate_shift = modulate_shift
        self.modulate_scale = modulate_scale
        self.is_last = is_last
        self.k=k

        self.linear = nn.Linear(in_size, out_size)

        if modulate_shift:
            self.modulate_shift_layer = nn.Linear(latent_modulation_dim, out_size)
            self.v_shift_layer = nn.Linear(latent_v_dim, out_size)
        if modulate_scale:
            self.modulate_scale_layer = nn.Linear(latent_modulation_dim, out_size)


        self._init(w0, is_first)

    def _init(self, w0, is_first):
        print('wo init',w0 )
        dim_in = self.linear.weight.size(1)
        w_std = 1/dim_in if is_first else (math.sqrt(6.0/dim_in)/w0)
      # w_finer = 1/dim_in# if is_first else 1
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        nn.init.uniform_(self.linear.bias, -w_std, -w_std)

    def forward(self, x, latent, v):
        x = self.linear(x)

        if not self.is_last:
            shift = 0.0 if not self.modulate_shift else self.modulate_shift_layer(latent)
            shift_v =  0.0 if not self.modulate_shift else self.v_shift_layer(v)
            scale = 1.0 if not self.modulate_scale else self.modulate_scale_layer(latent)
            if self.modulate_shift:
                if len(shift.shape) == 2:
                    shift = shift.unsqueeze(dim=1)
                    shift_v = shift_v.unsqueeze(dim=1)
            if self.modulate_scale:
                if len(scale.shape) == 2:
                    scale = scale.unsqueeze(dim=1)
            x = scale * x + shift + shift_v

           #if self.k<6:
               # x=self.w0 * x
              #  x=torch.exp(-x**2)*torch.sin(self.w0 * x)
           # else:

              #  x = torch.sin(self.w0 * x)
            x=self.w0 * x
            x=torch.sin(x)
           # print('x', x.max(), x.min())
           # x = torch.sin(self.w0*(torch.abs(x)+1) * x)
         #   print('x2', x.max(), x.min())
        return x


class LatentModulatedSIRENLayer_v5(nn.Module):
    def __init__(self, in_size, out_size, latent_modulation_dim: 512, latent_v_dim: 512, latent_s_dim: 512,w0=30.,
                 modulate_shift=True, modulate_scale=False, is_first=False, is_last=False, k=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.latent_modulation_dim = latent_modulation_dim
        self.latent_v_dim = latent_v_dim
        self.latent_s_dim = latent_s_dim

        self.w0 = w0
        self.modulate_shift = modulate_shift
        self.modulate_scale = modulate_scale
        self.is_last = is_last
        self.k=k

        self.linear = nn.Linear(in_size, out_size)

        if modulate_shift:
            self.modulate_shift_layer = nn.Linear(latent_modulation_dim, out_size)
            self.v_shift_layer = nn.Linear(latent_v_dim, out_size)
            self.s_shift_layer = nn.Linear(1, out_size)
        if modulate_scale:
            self.modulate_scale_layer = nn.Linear(latent_modulation_dim, out_size)


        self._init(w0, is_first)

    def _init(self, w0, is_first):
        dim_in = self.linear.weight.size(1)
        w_std = 1/dim_in if is_first else (math.sqrt(6.0/dim_in)/w0)
        w_finer = 1/dim_in# if is_first else 1
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        nn.init.uniform_(self.linear.bias, -w_std, -w_std)

    def forward(self, x, latent, v, s):
        x = self.linear(x)
        if not self.is_last:
            shift = 0.0 if not self.modulate_shift else self.modulate_shift_layer(latent)
            shift_v =  0.0 if not self.modulate_shift else self.v_shift_layer(v)
            shift_s =  0.0 if not self.modulate_shift else self.s_shift_layer(s)
            scale = 1.0 if not self.modulate_scale else self.modulate_scale_layer(latent)
            if self.modulate_shift:
                if len(shift.shape) == 2:
                    shift = shift.unsqueeze(dim=1)
                    shift_v = shift_v.unsqueeze(dim=1)
                    shift_s = shift_s.unsqueeze(dim=0)
            if self.modulate_scale:
                if len(scale.shape) == 2:
                    scale = scale.unsqueeze(dim=1)
            x = scale * x + shift + shift_v + shift_s[None,...]
            x=self.w0 * x
            x=torch.sin(x)

        return x


class LatentModulatedSIRENLayer_spatial(nn.Module):
    def __init__(self, in_size, out_size, latent_modulation_dim: 512, w0=30.,
                 modulate_shift=True, modulate_scale=False, is_first=False, is_last=False):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.latent_modulation_dim = latent_modulation_dim
        self.w0 = w0
        self.modulate_shift = modulate_shift
        self.modulate_scale = modulate_scale
        self.is_last = is_last
        

        self.linear = nn.Linear(in_size, out_size)

        if modulate_shift:
           # self.modulate_shift_layer = nn.Linear(latent_modulation_dim, out_size)
            self.modulate_shift_layer = nn.Conv2d(64, out_size, kernel_size=1)
            self.v_shift_layer = nn.Conv2d(64, out_size, kernel_size=1)
                        

  
        self._init(w0, is_first)

    def _init(self, w0, is_first):
        dim_in = self.linear.weight.size(1)
        w_std = 1/dim_in if is_first else (math.sqrt(6.0/dim_in)/w0)
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        nn.init.uniform_(self.linear.bias, -w_std, w_std)

    def get_nearest_patch_vectors(self, latent_tensor, coords):
        """
        Args:
        latent_tensor: torch.Tensor of shape [B, C, H, W] = [4, 64, 32, 32]
        coords: torch.Tensor of shape [B, N, 2], with values in [-1, 1]

        Returns:
            patch_vectors: torch.Tensor of shape [B, N, C] = [4, 2000, 64]
        """
        B, C, H, W = latent_tensor.shape  # [4, 64, 32, 32]
        _, N, _ = coords.shape            # [4, 2000, 2]

        x = coords[..., 0]  # [B, N]
        y = coords[..., 1]  # [B, N]

        # Normalize coordinates to [0, H-1] / [0, W-1] and round
        ix = ((x + 1) / 2 * (W - 1)).round().long()  # [B, N]
        iy = ((y + 1) / 2 * (H - 1)).round().long()  # [B, N]
        # Clamp to valid range
        ix = torch.clamp(ix, 0, W - 1)
        iy = torch.clamp(iy, 0, H - 1)

        # Prepare indexing tensors
        batch_idx = torch.arange(B).view(B, 1).expand(B, N)  # [B, N]

        # Use advanced indexing to extract the [C] vector for each (iy, ix)
        patch_vectors = latent_tensor[batch_idx, :, iy, ix]  # [B, C, N]
        
        # Transpose to [B, N, C]
        patch_vectors = patch_vectors

        return patch_vectors

    def forward(self, x, latent, v):
   
        
        x = self.linear(x)
        
        
        if not self.is_last:
            shift = 0.0 if not self.modulate_shift else self.modulate_shift_layer(latent)
            scale = 1.0 if not self.modulate_scale else self.modulate_scale_layer(latent)
            patch=self.get_nearest_patch_vectors(shift, x)
            
            shift_v =  0.0 if not self.modulate_shift else self.v_shift_layer(v)
            patch_v=self.get_nearest_patch_vectors(shift_v, x)


            if self.modulate_shift:
                if len(shift.shape) == 2:
                    shift = shift.unsqueeze(dim=1)

            if self.modulate_scale:
                if len(scale.shape) == 2:
                    scale = scale.unsqueeze(dim=1)
            x = scale * x + patch + patch_v
           
            x = torch.sin(self.w0 * x)
        return x





class LatentModulatedSIRENLayer_spatial_plain(nn.Module):
    def __init__(self, in_size, out_size, latent_modulation_dim: 512, w0=30.,
                 modulate_shift=True, modulate_scale=False, is_first=False, is_last=False):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.latent_modulation_dim = latent_modulation_dim
        self.w0 = w0
        self.modulate_shift = modulate_shift
        self.modulate_scale = modulate_scale
        self.is_last = is_last
        

        self.linear = nn.Linear(in_size, out_size)

        if modulate_shift:
           # self.modulate_shift_layer = nn.Linear(latent_modulation_dim, out_size)
            self.modulate_shift_layer = nn.Conv2d(64, out_size, kernel_size=1)
           
                        

  
        self._init(w0, is_first)

    def _init(self, w0, is_first):
        dim_in = self.linear.weight.size(1)
        w_std = 1/dim_in if is_first else (math.sqrt(6.0/dim_in)/w0)
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        nn.init.uniform_(self.linear.bias, -w_std, w_std)

    def get_nearest_patch_vectors(self, latent_tensor, coords):
        """
        Args:
        latent_tensor: torch.Tensor of shape [B, C, H, W] = [4, 64, 32, 32]
        coords: torch.Tensor of shape [B, N, 2], with values in [-1, 1]

        Returns:
            patch_vectors: torch.Tensor of shape [B, N, C] = [4, 2000, 64]
        """
        B, C, H, W = latent_tensor.shape  # [4, 64, 32, 32]
        _, N, _ = coords.shape            # [4, 2000, 2]

        x = coords[..., 0]  # [B, N]
        y = coords[..., 1]  # [B, N]

        # Normalize coordinates to [0, H-1] / [0, W-1] and round
        ix = ((x + 1) / 2 * (W - 1)).round().long()  # [B, N]
        iy = ((y + 1) / 2 * (H - 1)).round().long()  # [B, N]
        # Clamp to valid range
        ix = torch.clamp(ix, 0, W - 1)
        iy = torch.clamp(iy, 0, H - 1)

        # Prepare indexing tensors
        batch_idx = torch.arange(B).view(B, 1).expand(B, N)  # [B, N]

        # Use advanced indexing to extract the [C] vector for each (iy, ix)
        patch_vectors = latent_tensor[batch_idx, :, iy, ix]  # [B, C, N]
        
        # Transpose to [B, N, C]
        patch_vectors = patch_vectors

        return patch_vectors

    def forward(self, x, latent):
   
        
        x = self.linear(x)
        
        
        if not self.is_last:
            shift = 0.0 if not self.modulate_shift else self.modulate_shift_layer(latent)
            scale = 1.0 if not self.modulate_scale else self.modulate_scale_layer(latent)
            patch=self.get_nearest_patch_vectors(shift, x)
            
            if self.modulate_shift:
                if len(shift.shape) == 2:
                    shift = shift.unsqueeze(dim=1)

            if self.modulate_scale:
                if len(scale.shape) == 2:
                    scale = scale.unsqueeze(dim=1)
            x = scale * x + patch 
           
            x = torch.sin(self.w0 * x)
        return x
