import torch
from torch import nn

from models.layers import LatentModulatedSIRENLayer, LatentModulatedSIRENLayer_v3, LatentModulatedSIRENLayer_spatial, LatentModulatedSIRENLayer_spatial_plain

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LatentModulatedSIREN_spatial(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 hidden_size=256,
                 num_layers=5,
                 latent_modulation_dim=512,
                 w0=30.,
                 w0_increments=0.,
                 modulate_shift=True,
                 modulate_scale=False,
                 enable_skip_connections=True, 
                 guidance = False):
        super().__init__()
        layers = []
        for i in range(num_layers-1):
            is_first = i == 0
            layer_in_size = in_size if is_first else hidden_size
            layers.append(LatentModulatedSIRENLayer_spatial(in_size=layer_in_size, out_size=hidden_size,
                                                    latent_modulation_dim=latent_modulation_dim, w0=w0,
                                                    modulate_shift=modulate_shift, modulate_scale=modulate_scale,
                                                    is_first=is_first))
            w0 += w0_increments  # Allows for layer adaptive w0s
        self.layers = nn.ModuleList(layers)
        print('got spatial layers')
        self.last_layer = LatentModulatedSIRENLayer_spatial(in_size=hidden_size, out_size=out_size,
                                                    latent_modulation_dim=latent_modulation_dim, w0=w0,
                                                    modulate_shift=modulate_shift, modulate_scale=modulate_scale,
                                                    is_last=True)
        self.enable_skip_connections = enable_skip_connections
        self.modulations = torch.zeros(size=[1, 64, 4 , 4], requires_grad=True).to(device)  #spatial latent for one frame
        self.vdim = torch.zeros(size=[1, 64, 4 , 4], requires_grad=True).to(device) 
        self.guidance = guidance
        if self.guidance:
              
            self.simplecls = nn.Sequential(
                nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=2, stride=2),
                nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten(),
                nn.Linear(32, 1)
            )


    def reset_modulations(self):
        self.modulations = self.modulations.detach() * 0
        self.modulations.requires_grad = True
        
    def reset_vdim(self):
        self.vdim = self.vdim.detach() * 0
        self.vdim.requires_grad = True

    def forward(self, x, get_features=False):
        x = self.layers[0](x, self.modulations, self.vdim)
        for layer in self.layers[1:]:
            y = layer(x, self.modulations, self.vdim)
            if self.enable_skip_connections:
                x = x + y
            else:
                x = y
        features = x
        out = self.last_layer(features, self.modulations, self.vdim) + 0.5
        # if self.guidance:
        #     print('modulations', self.modulations.shape)
       
        #     output_reg = self.simplecls(self.modulations[None,...])
        #     print('outreg', output_reg.shape)
        #     return out, output_reg

        if get_features:
            return out, features
        else:
            return out



class LatentModulatedSIREN_spatial_plain(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 hidden_size=256,
                 num_layers=5,
                 latent_modulation_dim=512,
                 w0=30.,
                 w0_increments=0.,
                 modulate_shift=True,
                 modulate_scale=False,
                 enable_skip_connections=True, 
                 guidance = False):
        super().__init__()
        layers = []
        for i in range(num_layers-1):
            is_first = i == 0
            layer_in_size = in_size if is_first else hidden_size
            layers.append(LatentModulatedSIRENLayer_spatial_plain(in_size=layer_in_size, out_size=hidden_size,
                                                    latent_modulation_dim=latent_modulation_dim, w0=w0,
                                                    modulate_shift=modulate_shift, modulate_scale=modulate_scale,
                                                    is_first=is_first))
            w0 += w0_increments  # Allows for layer adaptive w0s
        self.layers = nn.ModuleList(layers)
        print('got spatial layers')
        self.last_layer = LatentModulatedSIRENLayer_spatial_plain(in_size=hidden_size, out_size=out_size,
                                                    latent_modulation_dim=latent_modulation_dim, w0=w0,
                                                    modulate_shift=modulate_shift, modulate_scale=modulate_scale,
                                                    is_last=True)
        self.enable_skip_connections = enable_skip_connections
        self.modulations = torch.zeros(size=[1, 64, 4 , 4], requires_grad=True).to(device)  #spatial latent for one frame
        self.vdim = torch.zeros(size=[1, 64, 4 , 4], requires_grad=True).to(device) 
        self.guidance = guidance
        if self.guidance:
              
            self.simplecls = nn.Sequential(
                nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=2, stride=2),
                nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten(),
                nn.Linear(32, 1)
            )


    def reset_modulations(self):
        self.modulations = self.modulations.detach() * 0
        self.modulations.requires_grad = True
        
   
    def forward(self, x, get_features=False):
        x = self.layers[0](x, self.modulations)
        for layer in self.layers[1:]:
            y = layer(x, self.modulations)
            if self.enable_skip_connections:
                x = x + y
            else:
                x = y
        features = x
        out = self.last_layer(features, self.modulations) + 0.5
        # if self.guidance:
        #     print('modulations', self.modulations.shape)
       
        #     output_reg = self.simplecls(self.modulations[None,...])
        #     print('outreg', output_reg.shape)
        #     return out, output_reg

        if get_features:
            return out, features
        else:
            return out

class LatentModulatedSIREN_v2(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 hidden_size=256,
                 num_layers=5,
                 latent_modulation_dim=512,
                 v_dim = 512,
                 w0=30.,
                 w0_increments=0.,
                 modulate_shift=True,
                 modulate_scale=False,
                 enable_skip_connections=True,
                 mode = 'concat'):
        super().__init__()
        layers = []
        print('vdim2', v_dim)
        self.vdim2 = v_dim
        self.mode = mode
        for i in range(num_layers-1):
            is_first = i == 0
            layer_in_size = in_size if is_first else hidden_size
            if v_dim >0:
                if self.mode == 'additive':
                    latentsize = latent_modulation_dim
                else:
                     latentsize = latent_modulation_dim + v_dim   #concatenate vector
                

            else:
                latentsize = latent_modulation_dim
                self.vdim = 0
              
            layers.append(LatentModulatedSIRENLayer(in_size=layer_in_size, out_size=hidden_size,
                                                    latent_modulation_dim=latentsize, w0=w0,
                                                    modulate_shift=modulate_shift, modulate_scale=modulate_scale,
                                                    is_first=is_first))
            w0 += w0_increments  # Allows for layer adaptive w0s
        self.layers = nn.ModuleList(layers)
        self.last_layer = LatentModulatedSIRENLayer(in_size=hidden_size, out_size=out_size,
                                                   latent_modulation_dim=latentsize, w0=w0,
                                                    modulate_shift=modulate_shift, modulate_scale=modulate_scale,
                                                    is_last=True)
        self.enable_skip_connections = enable_skip_connections
        self.modulations = torch.zeros(size=[latent_modulation_dim], requires_grad=True).to(device)
        self.vdim = torch.zeros(size=[1,latent_modulation_dim], requires_grad=True).to(device)

        

    def reset_modulations(self):
        self.modulations = self.modulations.detach() * 0
        self.modulations.requires_grad = True
    
    def reset_vdim(self):
        self.vdim = self.vdim.detach() * 0
        self.vdim.requires_grad = True

    def forward(self, x, get_features=False):
        
        
        if self.vdim2 == 0:

            concat = self.modulations
            
        else: 
            if self.mode =='concat':
                  concat =  torch.cat((self.modulations, self.vdim.repeat(self.modulations.shape[0],1)), dim=1)
                  
            else:
                concat = self.modulations + self.vdim
        
        x = self.layers[0](x, concat)
        for layer in self.layers[1:]:
            y = layer(x, concat)
            if self.enable_skip_connections:
                x = x + y
            else:
                x = y
        features = x
        out = self.last_layer(features, concat) + 0.5   #why is there a +0.5?
        
        if get_features:
            return out, features
        else:
            return out



class LatentModulatedSIREN_v3(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 hidden_size=256,
                 num_layers=5,
                 latent_modulation_dim=512,
                 v_dim = 512,
                 w0=30.,
                 w0_increments=0.,
                 modulate_shift=True,
                 modulate_scale=False,
                 enable_skip_connections=True,
                 mode = 'concat',
                 num_frames = 60,
                 guidance = False):
        super().__init__()
        layers = []
        print('vdim2', v_dim)
        self.vdim2 = v_dim
        self.mode = mode
        self.num_frames = num_frames
        self.guidance = guidance
        for i in range(num_layers-1):
            is_first = i == 0
            layer_in_size = in_size if is_first else hidden_size
            if v_dim >0:
                if self.mode == 'separate':
                    latentsize = latent_modulation_dim
                    latentsize_v = v_dim
                else:
                     latentsize = latent_modulation_dim  #concatenate vector
                

            else:
                latentsize = latent_modulation_dim
                self.vdim = 0
              
            layers.append(LatentModulatedSIRENLayer_v3(in_size=layer_in_size, out_size=hidden_size,
                                                    latent_modulation_dim=latentsize, latent_v_dim= latentsize_v, w0=w0,
                                                    modulate_shift=modulate_shift, modulate_scale=modulate_scale,
                                                    is_first=is_first, k=i))
            w0 += w0_increments  # Allows for layer adaptive w0s
        self.layers = nn.ModuleList(layers)
        self.last_layer = LatentModulatedSIRENLayer_v3(in_size=hidden_size, out_size=out_size,
                                                   latent_modulation_dim=latentsize, latent_v_dim= latentsize_v, w0=w0,
                                                    modulate_shift=modulate_shift, modulate_scale=modulate_scale,
                                                    is_last=True)
        self.enable_skip_connections = enable_skip_connections
        self.modulations = torch.zeros(size=[latent_modulation_dim], requires_grad=True).to(device)
        self.vdim = torch.zeros(size=[1,latent_modulation_dim], requires_grad=True).to(device)
        if self.guidance:
            self.simplecls =  nn.Sequential(
            nn.Flatten(),
            nn.Linear(latentsize*self.num_frames, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
        )
        

    def reset_modulations(self):
        self.modulations = self.modulations.detach() * 0
        self.modulations.requires_grad = True
    
    def reset_vdim(self):
        self.vdim = self.vdim.detach() * 0
        self.vdim.requires_grad = True

    def forward(self, x, get_features=False):
        
        
   
        x = self.layers[0](x, self.modulations, self.vdim)
        for layer in self.layers[1:]:
            y = layer(x,  self.modulations, self.vdim)
            if self.enable_skip_connections:
                x = x + y
            else:
                x = y
        features = x
        out = self.last_layer(features, self.modulations, self.vdim) + 0.5   #why is there a +0.5?
     #   if self.guidance:
         #   print('modulations', self.modulations.shape)
        #    flat = torch.flatten(self.modulations)
        #    print('flat', flat.shape)
         #   output_reg = self.simplecls(flat[None,...])
         #   print('outreg', output_reg)
        #    return out, output_reg
            
        if get_features:
            return out, features
        else:
            return out





class LatentModulatedSIREN_v4(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 hidden_size=256,
                 num_layers=5,
                 latent_modulation_dim=512,
                 v_dim =0,
                 w0=30.,
                 w0_increments=0.,
                 modulate_shift=True,
                 modulate_scale=False,
                 enable_skip_connections=True,
                 mode = 'plain'):
        super().__init__()
        layers = []
        for i in range(num_layers-1):
            is_first = i == 0
            layer_in_size = in_size if is_first else hidden_size
            layers.append(LatentModulatedSIRENLayer(in_size=layer_in_size, out_size=hidden_size,
                                                    latent_modulation_dim=latent_modulation_dim, w0=w0,
                                                    modulate_shift=modulate_shift, modulate_scale=modulate_scale,
                                                    is_first=is_first))
            w0 = w0+ w0_increments  # Allows for layer adaptive w0s
        self.layers = nn.ModuleList(layers)
        self.time_layer = nn.Sequential(nn.Linear(1,256), nn.ReLU(), nn.Linear(256, latent_modulation_dim))
      #  self.time_layer =nn.Linear(1,latent_modulation_dim)

        self.vdim = 0
        self.mode = mode
        self.last_layer = LatentModulatedSIRENLayer(in_size=hidden_size, out_size=out_size,
                                                    latent_modulation_dim=latent_modulation_dim, w0=w0,
                                                    modulate_shift=modulate_shift, modulate_scale=modulate_scale,
                                                    is_last=True)
        self.enable_skip_connections = enable_skip_connections
     #   self.modulations = torch.zeros(size=[latent_modulation_dim], requires_grad=True).to(device)

    def reset_modulations(self):
       
          #  self.time_layer.weight.data.zero_()
          #  self.time_layer.bias.data.zero_()
          #  self.time_layer.reset_parameters()
        with torch.no_grad():
              for param in self.time_layer.parameters():
                    param.zero_()


     

    def forward(self, x, t, get_features=False):
        t=t.permute(1,0,2)
        self.modulations = self.time_layer(t)[:,0,:]
        x = self.layers[0](x, self.modulations)
        for layer in self.layers[1:]:
            y = layer(x, self.modulations)
            if self.enable_skip_connections:
                x = x + y
            else:
                x = y
        features = x
        out = self.last_layer(features, self.modulations) + 0.5
        if get_features:
            return out, features
        else:
            return out


