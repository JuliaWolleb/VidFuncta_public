import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from einops import rearrange
import matplotlib.pyplot as plt

def exists(val):
    return val is not None


class ModelWrapper(nn.Module):
    def __init__(self, args, model):
        super().__init__()
        self.args = args
        self.model = model
        self.data_type = args.data_type
        print('datatype', self.data_type) 
        self.comment = args.comment

        self.sampled_coord = None
        self.sampled_index = None
        self.gradncp_coord = None
        self.gradncp_index = None
        device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
        args.device = device
        self.mode = args.mode
       # args.data_type = 'img'
        args.img_size = 112
        args.data_size = (1, args.img_size, args.img_size)
        if args.dimension == '3d':
             args.data_size = (1, args.num_frames, args.img_size, args.img_size)

        if self.data_type == 'img':
            self.width = args.data_size[1]
            self.height = args.data_size[2]

            mgrid = self.shape_to_coords((self.width, self.height))
            mgrid = rearrange(mgrid, 'h w c -> (h w) c')

        elif self.data_type == 'img3d':
            print('got 3d image')
            self.width = args.data_size[1]
            self.height = args.data_size[2]
            self.depth = args.data_size[3]

            mgrid = self.shape_to_coords((self.width, self.height, self.depth))
            mgrid = rearrange(mgrid, 'h w d c -> (h w d) c')
            print('mgrdig', mgrid.shape)

        elif self.data_type == 'timeseries':
            self.length = args.data_size[-1]
            mgrid = self.shape_to_coords([self.length])

        else:
            raise NotImplementedError()

        self.register_buffer('grid', mgrid)

    def coord_init(self):
        self.sampled_coord = None
        self.sampled_index = None
        self.gradncp_coord = None
        self.gradncp_index = None

    def get_batch_coords(self, x=None):
        if x is None:
            meta_batch_size = 1
        else:
            meta_batch_size = x.size(0)

        # batch of coordinates
        if self.sampled_coord is None and self.gradncp_coord is None:
            coords = self.grid
        elif self.gradncp_coord is not None:
            return self.gradncp_coord, meta_batch_size
        else:
            coords = self.sampled_coord
        coords = coords.clone().detach()[None, ...].repeat((meta_batch_size,) + (1,) * len(coords.shape))
        return coords, meta_batch_size

    def shape_to_coords(self, spatial_dims):
        coords = []
        for i in range(len(spatial_dims)):
            coords.append(torch.linspace(-1.0, 1.0, spatial_dims[i]))
        return torch.stack(torch.meshgrid(*coords, indexing='ij'), dim=-1)

    def sample_coordinates(self, sample_type, data):
        if sample_type == 'random':
            self.random_sample()
        elif sample_type == 'gradncp':
            if random.random() < 0.5:
                self.gradncp(data)
            else:
                self.random_sample()
        else:
            raise NotImplementedError()

    def gradncp(self, x):
        ratio = self.args.data_ratio
        meta_batch_size = x.size(0)
        coords = self.grid
        coords = coords.clone().detach()[None, ...].repeat((meta_batch_size,) + (1,) * len(coords.shape))
        coords = coords.to(self.args.device)
        with torch.no_grad():
            out, feature = self.model(coords, get_features=True)

        if self.data_type == 'img':
            out = rearrange(out, 'b hw c -> b c hw')
            feature = rearrange(feature, 'b hw f -> b f hw')
            x = rearrange(x, 'b c h w -> b c (h w)')
        elif self.data_type == 'img3d':
            out = rearrange(out, 'b hwd c -> b c hwd')
            feature = rearrange(feature, 'b hwd f -> b f hwd')
            x = rearrange(x, 'b c h w d -> b c (h w d)')
        elif self.data_type == 'timeseries':
            out = rearrange(out, 'b l c -> b c l')
            feature = rearrange(feature, 'b l f -> b f l')
        else:
            raise NotImplementedError()

        error = x - out

        gradient = -1 * feature.unsqueeze(dim=1) * error.unsqueeze(dim=2)
        gradient_bias = -1 * error.unsqueeze(dim=2)
        gradient = torch.cat([gradient, gradient_bias], dim=2)
        gradient = rearrange(gradient, 'b c f hw -> b (c f) hw')
        gradient_norm = torch.norm(gradient, dim=1)

        coords_len = gradient_norm.size(1)

        self.gradncp_index = torch.sort(gradient_norm, dim=1, descending=True)[1][:, :int(coords_len * ratio)]
        self.gradncp_coord = torch.gather(coords, 1, self.gradncp_index.unsqueeze(dim=2).repeat(1, 1, self.args.in_size))
        self.gradncp_index = self.gradncp_index.unsqueeze(dim=1).repeat(1, self.args.out_size, 1)

    def random_sample(self):
        coord_size = self.grid.size(0)
        #print('coord', coord_size)
        perm = torch.randperm(coord_size)
        self.sampled_index = perm[:int(self.args.data_ratio * coord_size)]
        self.sampled_coord = self.grid[self.sampled_index]
        return self.sampled_coord

    def forward(self, x=None, t=torch.zeros(1)):
        if self.data_type == 'img':
            if self.mode == 'time':
                return self.forward_img(x,t)
            elif self.comment == 'separate2':
             try:
                image = x['vid']
                label = x['label']
             except: 
                  image = None
                  label = t.to(self.args.device).float()
             return self.forward_img(image, label)
            else:
                return self.forward_img(x)
        if self.data_type == 'img3d':
            return self.forward_img3d(x)
        if self.data_type == 'timeseries':
            return self.forward_timeseries(x)
        else:
            raise NotImplementedError()

    def forward_img(self, x, t=None):
        coords, meta_batch_size = self.get_batch_coords(x)
        coords = coords.to(self.args.device)
        if self.mode == 'time':
           
            t = t.to(self.args.device) 
            out = self.model(coords,t)
        elif self.comment == 'separate2':
            t = t.to(self.args.device)
            out = self.model(coords,t)
            print('out', out.shape)
        else:
            out, features = self.model(coords, get_features = True)
            # if features.shape[1] == 112*112:
            #     print('f0', features.shape)
            #     features = rearrange(features, 'b (h w) f -> b f h w', h=112, w=112)
            #  #   features = rearrange(features, 'b hw f -> b f h w')
            #     print('features', features.shape)
            #     selected_maps_1 = features[0, :12]  # Shape: [12, 112, 112]
            #     selected_maps_2 = features[1, :12]  # Shape: [12, 112, 112]
            #     # Create a figure with 3 rows × 4 columns
            #     plt.figure(1)
            #     fig, axes = plt.subplots(3, 4, figsize=(12, 9))

            #     for i, ax in enumerate(axes.flat):
            #         fmap = selected_maps_1[i].cpu().detach().numpy()
            #         ax.imshow(fmap, cmap='viridis')
            #         ax.axis('off')
            #         ax.set_title(f'Feature {i+1}')

            #     plt.tight_layout()
            #     plt.savefig('feature_maps_untrained.png', dpi=300)
            #     plt.close()
        compute_kl=False
        if compute_kl==True:
                kl_div =  -0.5 * torch.sum(1 + self.model.modulations_std - self.model.modulations_mean.pow(2) - self.model.modulations_std.exp(), dim=1)
                

        out = rearrange(out, 'b hw c -> b c hw')
        if exists(x):
            if self.sampled_coord is None and self.gradncp_coord is None:
                return F.mse_loss(x.view(meta_batch_size, -1), out.reshape(meta_batch_size, -1), reduce=False).mean(dim=1)
            elif self.gradncp_coord is not None:
                x = rearrange(x, 'b c h w -> b c (h w)')
                x = torch.gather(x, 2, self.gradncp_index)
                return F.mse_loss(x.view(meta_batch_size, -1), out.reshape(meta_batch_size, -1), reduce=False).mean(dim=1)
            else:
                x = rearrange(x, 'b c h w -> b c (h w)')[:, :, self.sampled_index]
                mseloss = F.mse_loss(x.view(meta_batch_size, -1), out.reshape(meta_batch_size, -1), reduce=False).mean(dim=1)
             #   print('losses', mseloss, kl_div)
                return mseloss# + 0.01*kl_div
        
        out = rearrange(out, 'b c (h w) -> b c h w', h=self.height, w=self.width)
        return out

    def forward_img3d(self, x):
        coords, meta_batch_size = self.get_batch_coords(x)
        coords = coords.to(self.args.device)

        out = self.model(coords)
        out = rearrange(out, 'b whd c -> b c whd')

        if exists(x):
            if self.sampled_coord is None and self.gradncp_coord is None:
                return F.mse_loss(x.view(meta_batch_size, -1), out.reshape(meta_batch_size, -1), reduce=False).mean(dim=1)
            elif self.gradncp_coord is not None:
                x = rearrange(x, 'b c w h d -> b c (w h d)')
                x = torch.gather(x, 2, self.gradncp_index)
                return F.mse_loss(x.view(meta_batch_size, -1), out.reshape(meta_batch_size, -1), reduce=False).mean(dim=1)
            else:
                x = rearrange(x, 'b c w h d -> b c (w h d)')[:, :, self.sampled_index]
                return F.mse_loss(x.view(meta_batch_size, -1), out.reshape(meta_batch_size, -1), reduce=False).mean(dim=1)
  

        out = rearrange(out, 'b c (w h d) -> b c w h d', h=self.height, w=self.width, d=self.depth)  #should be (1,1,10,112,112)

        return out

    def forward_timeseries(self, x):
        coords, meta_batch_size = self.get_batch_coords(x)
        coords = coords.to(self.args.device)

        out = self.model(coords)
        out = rearrange(out, 'b l c -> b c l')

        if exists(x):
            if self.sampled_coord is None and self.gradncp_coord is None:
                return F.mse_loss(x.view(meta_batch_size, -1), out.reshape(meta_batch_size, -1), reduce=False).mean(dim=1)
            elif self.gradncp_coord is not None:
                x = torch.gather(x, 2, self.gradncp_index)
                return F.mse_loss(x.view(meta_batch_size, -1), out.reshape(meta_batch_size, -1), reduce=False).mean(dim=1)
            else:
                x = x[:, :, self.sampled_index]
                return F.mse_loss(x.view(meta_batch_size, -1), out.reshape(meta_batch_size, -1), reduce=False).mean(dim=1)
       
        return out