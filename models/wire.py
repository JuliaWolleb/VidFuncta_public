
import os
import sys

import pdb

import numpy as np
import torch
from torch import nn

import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RealGaborLayer(nn.Module):
    '''
        Implicit representation with Gabor nonlinearity
        
        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega: Frequency of Gabor sinusoid term
            scale: Scaling of Gabor Gaussian term
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=10.0, modulate_shift = True, latent_modulation_dim= 2048,
                 trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        self.modulate_shift = modulate_shift

        
        self.in_features = in_features
        if modulate_shift:
            self.modulate_shift_layer = nn.Linear(latent_modulation_dim, out_features)
        
        self.freqs = nn.Linear(in_features, out_features, bias=bias)
        self.scale = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, input, latent):
        omega = self.omega_0 * self.freqs(input)
        scale = self.scale(input) * self.scale_0
        shift = 0.0 if not self.modulate_shift else self.modulate_shift_layer(latent)
        if self.modulate_shift:
                if len(shift.shape) == 2:
                    shift = shift.unsqueeze(dim=1)
        omega = omega + shift
        
        return torch.cos(omega)*torch.exp(-(scale**2))

class ComplexGaborLayer(nn.Module):
    '''
        Implicit representation with complex Gabor nonlinearity
        
        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega0: Frequency of Gabor sinusoid term
            sigma0: Scaling of Gabor Gaussian term
            trainable: If True, omega and sigma are trainable parameters
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=40.0, latent_modulation_dim= 2048,
                 trainable=False, modulate_shift = True, is_last=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        self.is_last = is_last
        self.modulate_shift = modulate_shift
        self.latent_modulation_dim = latent_modulation_dim


        
        self.in_features = in_features
        
        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat
            
        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0*torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0*torch.ones(1), trainable)
        
        self.linear = nn.Linear(in_features,
                                out_features,
                                bias=bias,
                                dtype=dtype)

        if modulate_shift:
            self.modulate_shift_layer = nn.Linear(latent_modulation_dim, out_features)
    
    def forward(self, input, latent):
       
        if not self.is_last:
            shift = 0.0 if not self.modulate_shift else self.modulate_shift_layer(latent)
            if self.modulate_shift:
                if len(shift.shape) == 2:
                    shift = shift.unsqueeze(dim=1)
           

        lin = self.linear(input)
        omega = self.omega_0 * (lin + shift)
        scale = self.scale_0 * (lin + shift)
     
        
        return torch.exp(1j*omega - scale.abs().square())
    
class INR(nn.Module):
    def __init__(self, in_features=2, hidden_features=256, 
                 hidden_layers=10, 
                 out_features=1, outermost_linear=True, latent_modulation_dim=2048,
                 first_omega_0=30, hidden_omega_0=30., scale=10.0,
                 pos_encode=False, sidelength=512, fn_samples=None,
                 use_nyquist=True, enable_skip_connections=True):
        super().__init__()
        
        # All results in the paper were with the default complex 'gabor' nonlinearity
        self.nonlin =  ComplexGaborLayer
        
        # Since complex numbers are two real numbers, reduce the number of 
        # hidden parameters by 2
        hidden_features = int(hidden_features/np.sqrt(2))
        dtype = torch.cfloat
        self.complex = True
        self.wavelet = 'gabor'    
        self.enable_skip_connections = enable_skip_connections

        
        # Legacy parameter
        self.pos_encode = False
            
        self.net = []

        #in_features is in_size, should be 2
       # hidden_features is hidden_size, should be 256
       #out_features is out_size, should be 1
        self.net.append(self.nonlin(in_features,
                                    hidden_features, 
                                    omega0=first_omega_0,
                                    sigma0=scale,
                                    is_first=True,
                                    trainable=False))

        for i in range(hidden_layers):


            self.net.append(self.nonlin(hidden_features,
                                        hidden_features, 
                                        omega0=hidden_omega_0,
                                        sigma0=scale))

        self.final_linear = nn.Linear(hidden_features,
                                 out_features,
                                 dtype=dtype)            
        #self.net.append(final_linear)
        
        self.net = nn.Sequential(*self.net)
        self.modulations = torch.zeros(size=[latent_modulation_dim], requires_grad=True).to(device)

    def reset_modulations(self):
        self.modulations = self.modulations.detach() * 0
        self.modulations.requires_grad = True

    def forward(self, x):
        x = self.net[0](x, self.modulations)
        for layer in self.net[1:]:
            y = layer(x, self.modulations)
            if self.enable_skip_connections:
                x = x + y
            else:
                x = y

        output = self.final_linear(x)
        
        if self.wavelet == 'gabor':
            return output.real
         
        return output