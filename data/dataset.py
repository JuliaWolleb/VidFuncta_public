import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
import pandas as pd
import os
import nibabel
import cv2
from torch.utils.data import DataLoader, Dataset
from imutils import paths
import random
import sys
import matplotlib.pyplot as plt 
import torch.nn.functional as F

fourcc = cv2.VideoWriter_fourcc(*'mp4v')


def combine_class(c):
    if c > 1:
        return c - 1
    else:
        return c


class JPGDataset(Dataset):
    def __init__(self, folder_path = '/gpfs/gibbs/project/hartley/jw3234/medfuncta/vidfuncta/data/MRI', img_size = 112, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = [
            os.path.join(folder_path, fname)
            for fname in os.listdir(folder_path)
            if fname.lower().endswith('.jpg')
        ]
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
       # image = Image.open(self.image_paths[idx]).convert('RGB')
            image = cv2.imread(self.image_paths[idx])
            image = cv2.resize(image, (112, 112))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            out = np.array(image)

             # Clip and normalize the images
            out_clipped = np.clip(out, np.quantile(out, 0.001), np.quantile(out, 0.999))
            out_normalized = (out_clipped - np.min(out_clipped)) / (np.max(out_clipped) - np.min(out_clipped))
            out = torch.tensor(out_normalized)[None,...]
            # Insert dummy label
          
            label = 1

            # Downsampling

            # if self.img_size == 112:
            #     downsample = nn.AvgPool2d(kernel_size=2, stride=2)
            #     image = downsample(image)

            # if self.transform:
            #     image = self.transform(image)
            image = out
            print('image', image.shape)

            return image, label



class ECG1D(torch.utils.data.Dataset):
    def __init__(self, directory, test=False):
        super().__init__()

        if not test:
            self.df = pd.read_csv(directory + '/mitbih_train.csv')
        else:
            self.df = pd.read_csv(directory + '/mitbih_test.csv')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx, :-1].values.astype(float)
        label = self.df.iloc[idx, -1]
        sample = torch.tensor(sample, dtype=torch.float32).unsqueeze(dim=0)
        label = torch.tensor(label, dtype=torch.long)

        return sample, label


class BRATSVolumes(torch.utils.data.Dataset):
    def __init__(self, directory,  normalize=None, img_size=32):
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.normalize = normalize or (lambda x: x)
        self.img_size = img_size
        self.seqtypes = ['t1n', 't1c', 't2w', 't2f', 'seg']
        self.seqtypes_set = set(self.seqtypes)
        self.database = []

        for root, dirs, files in os.walk(self.directory):
            # Ensure determinism
            dirs.sort()
            files.sort()
            # if there are no subdirs, we have a datadir
            if not dirs:
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    seqtype = f.split('-')[4].split('.')[0]
                    datapoint[seqtype] = os.path.join(root, f)
                self.database.append(datapoint)

    def __getitem__(self, x):
        filedict = self.database[x]
        name = filedict['t1n']
        nib_img = nibabel.load(name)  # We only use t1 weighted images
        out = nib_img.get_fdata()

        # Clip and normalize the images
        out_clipped = np.clip(out, np.quantile(out, 0.001), np.quantile(out, 0.999))
        out_normalized = (out_clipped - np.min(out_clipped)) / (np.max(out_clipped) - np.min(out_clipped))
        out = torch.tensor(out_normalized)

        # Zero pad images
        image = torch.zeros(1, 256, 256, 256)
        image[:, 8:-8, 8:-8, 50:-51] = out

        # Downsampling
        if self.img_size == 32:
            downsample = nn.AvgPool3d(kernel_size=8, stride=8)
            image = downsample(image)

        # Normalization
        image = self.normalize(image)

        # Insert dummy label
        label = 1

        return image, label

    def __len__(self):
        return len(self.database)


class LIDCVolumes(torch.utils.data.Dataset):
    def __init__(self, directory, normalize=None, img_size=32):
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.normalize = normalize or (lambda x: x)
        self.img_size = img_size
        self.database = []

        for root, dirs, files in os.walk(self.directory):
            # Ensure determinism
            dirs.sort()
            files.sort()
            # if there are no subdirs, we have a datadir
            if not dirs:
                datapoint = dict()
                for f in files:
                    datapoint['image'] = os.path.join(root, f)
                if len(datapoint) != 0:
                    self.database.append(datapoint)

    def __getitem__(self, x):
        filedict = self.database[x]
        name = filedict['image']
        nib_img = nibabel.load(name)
        out = nib_img.get_fdata()

        out = torch.Tensor(out)

        image = torch.zeros(1, 256, 256, 256)
        image[:, :, :, :] = out

        if self.img_size == 32:
            downsample = nn.AvgPool3d(kernel_size=8, stride=8)
            image = downsample(image)

        # normalization
        image = self.normalize(image)

        # Insert dummy label
        label = 1

        return image, label

    def __len__(self):
        return len(self.database)




class LUS(torch.utils.data.Dataset):

        def __init__(self, datapath= '/home/julia/pocovid/data/image_dataset_blue', labels = '/home/julia/pocovid/data/severity_path.csv', img_size=64, transform=None):
            self.datapath= datapath
            self.labels = labels
            self.transform = transform
            self.image_paths = list(paths.list_images( datapath))
            self.img_size = img_size

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            
          #  print('path 0', self.image_paths[idx])
            image = cv2.imread(self.image_paths[idx])
            image = cv2.resize(image, (256, 256))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            out = np.array(image)
        
            

             # Clip and normalize the images
            out_clipped = np.clip(out, np.quantile(out, 0.001), np.quantile(out, 0.999))
            out_normalized = (out_clipped - np.min(out_clipped)) / (np.max(out_clipped) - np.min(out_clipped))
            out = torch.tensor(out_normalized)
            # Insert dummy label

            path_parts = self.image_paths[idx].split(os.path.sep)
   
            # extract the class label from the filename
            label = path_parts[-2]
            
            label_dict= {"cov": 0, "pne": 1, "reg": 2}

           
            label = 1
           # label =torch.tensor(self.labels[idx]).float()
         

            # Zero pad images
            image = torch.zeros(1, 256, 256)
            image[0, ...] = out

            # Downsampling
            if self.img_size == 64:
                downsample = nn.AvgPool2d(kernel_size=4, stride=4)
                image = downsample(image)

            elif self.img_size == 128:
                downsample = nn.AvgPool2d(kernel_size=2, stride=2)
                image = downsample(image)
            
           
          #  onehot =label#F.one_hot(label.long(), config.num_classes)


            if self.transform:
                
                image = self.transform(image)

            return image, label
        
        print('got loaders')




class LUS_video(torch.utils.data.Dataset):

        def __init__(self, datapath= '/home/julia/pocovid/data/pocus_videos/convex', labelsfile = '/home/julia/pocovid/data/severity_path.csv', img_size=32, transform=None, clip = 32):
           
            self.datapath= datapath
            self.labels = labelsfile
            self.transform = transform
            self.image_paths = [os.path.join(self.datapath, f) for f in os.listdir(self.datapath)]
            self.img_size = img_size
            self.clip = clip


        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            
            sys.exit('got wrong dataloader')
            curr_vid_path= self.image_paths[idx]
           
            filename = curr_vid_path.split('/')[-1]
            try:
                classname = filename.split('_')[0]
                
                label_dict= {"Cov": 0, "Pneu": 1, "Reg": 2, "Vir": 1, "pneu": 1}
                label=label_dict[classname]
            except: 
                classname = filename.split('-')[0]
                label_dict= {"Cov": 0, "Pneu": 1, "Reg": 2, "pneu": 1}
                label=label_dict[classname]
               
         
            cap = cv2.VideoCapture(curr_vid_path)
          
            current_data = []
            
            while cap.isOpened():
                frame_id = cap.get(1)
                ret, frame = cap.read()
                if (ret != True):
                    break
              
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (self.img_size, self.img_size))
                numpyimage = np.asarray(image)
               
                current_data.append(numpyimage)
               
            cap.release()
        
            out= np.asarray(current_data)

            if out.shape[0] > self.clip:  
                lapse = out.shape[0] -self.clip
                t= torch.randint(0, lapse, (1,)).item()
                out=out[t:t+self.clip,...]
          
        
             # Clip and normalize the images
            out_clipped = np.clip(out, np.quantile(out, 0.001), np.quantile(out, 0.999))
            out_normalized = (out_clipped - np.min(out_clipped)) / (np.max(out_clipped) - np.min(out_clipped))
            out = torch.tensor(out_normalized)
           
            # Insert dummy label

   
            # extract the class label from the filename
            
    
          
            # Zero pad videos
           
            vid = torch.zeros(self.clip,self.img_size,self.img_size)
            vid[:out.shape[0], ...] = out
          # vid = vid[None,...]
            

            # Downsampling
           # if self.img_size == 64:
            #    downsample = nn.AvgPool2d(kernel_size=4, stride=4)
            #    image = downsample(image)
           
          #  onehot =label#F.one_hot(label.long(), config.num_classes)


            if self.transform:  #TODO: change to video_transform to extract a clip
                
                out= self.transform(out)

            return  vid, label
        
        print('got loaders')


class LUS_video_blue_autoregressive(torch.utils.data.Dataset):
        def __init__(self, datapath= '/home/julia/COVID-BLUES/lus_videos', labelsfile = '/home/julia/COVID-BLUES/severity_path.csv', img_size=128, transform=None, clip = 32, num_frames = 32):
           
            self.datapath= datapath
            self.transform = transform
            self.image_paths = [os.path.join(self.datapath, f) for f in os.listdir(self.datapath)]
            self.img_size = img_size
            self.clip = clip
            self.num_frames = num_frames
            labels_df = pd.read_csv(labelsfile, index_col=0)
            try:
                labels_df.dropna(subset=['B_lines'], inplace=True)
            except:
                pass
            indices = labels_df.index
            labels = labels_df['B_lines'].values.flatten()
            self.labels_dict = dict(zip(indices, labels))
            path = labels_df['path']
            self.path_dict = dict(zip(indices, path))
        def __len__(self):
            return len(self.image_paths)
        def __getitem__(self, idx):
            label=self.labels_dict[idx]
            image_names=self.path_dict[idx]
            filename = image_names.split('/',4)[-1]
            curr_vid_path= os.path.join( self.datapath,  filename)
            
            cap = cv2.VideoCapture(curr_vid_path)
           
            current_data = []
            while cap.isOpened():
                frame_id = cap.get(1)
                ret, frame = cap.read()
                
                if (ret != True):

                    break
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (self.img_size, self.img_size))
                numpyimage = np.asarray(image)
            
                current_data.append(numpyimage)
            cap.release()
            out= np.asarray(current_data)
           
           
            if out.shape[0] > self.clip:
                lapse = out.shape[0] -self.clip
                t= torch.randint(0, lapse, (1,)).item()
                out=out[t:t+self.clip,...]

             # Clip and normalize the images
            out_clipped = np.clip(out, np.quantile(out, 0.001), np.quantile(out, 0.999))
            out_normalized = (out_clipped - np.min(out_clipped)) / (np.max(out_clipped) - np.min(out_clipped))
            out = torch.tensor(out_normalized)
            vid = torch.zeros(self.clip,self.img_size,self.img_size)
            vid[:out.shape[0], ...] = out
          
            
            if self.num_frames <60:
                
                selected=random.sample(range(1, self.clip), self.num_frames)
                vid =vid[selected,...]
            elif self.num_frames == 30:
                 vid = vid[ ::2, ...]
                 print('got only every second channel')
            else: 
                print('got all frames', vid.shape)
            return  vid, label
        print('got loaders')


class LUS_video_blue(torch.utils.data.Dataset):

        def __init__(self, datapath= './data/lus_videos', labelsfile = './data/severity_path.csv', img_size=128, transform=None, clip = 'all', num_frames = 4, selection = True, sobel = False):
            # /home/julia/pocovid/data/pocus_videos/convex
            self.datapath= datapath
            
            self.transform = transform
         #   self.image_paths = list(paths.list_images( datapath))
            self.image_paths = [os.path.join(self.datapath, f) for f in os.listdir(self.datapath)]
            print('paths', self.image_paths)
            print('len', len(self.image_paths))
            self.img_size = img_size
            self.clip = clip
            self.num_frames = num_frames
            self.selection = selection
            labels_df = pd.read_csv(labelsfile, index_col=0)
            try:
                labels_df.dropna(subset=['Severity_Score'], inplace=True)
            except:
                pass

            indices = labels_df.index
            
            labels = labels_df['Severity_Score'].values.flatten()
            
            self.labels_dict = dict(zip(indices, labels))
           
           
            path = labels_df['path']
            self.path_dict = dict(zip(indices, path))

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            label=self.labels_dict[idx]
            image_names=self.path_dict[idx]
            filename = image_names.split('/',4)[-1]
            curr_vid_path= os.path.join( self.datapath,  filename)
           
            cap = cv2.VideoCapture(curr_vid_path)
            fps = cap.get(cv2.CAP_PROP_FPS)

          
            current_data = []
            
            while cap.isOpened():
                frame_id = cap.get(1)

                ret, frame = cap.read()
                if (ret != True):
                    break
                image = cv2.resize(frame, (224,224))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
               # plt.figure()
              #  plt.imshow(image, cmap='gray')  # Tensor format: [batch, channel, height, width]
  

                numpyimage = np.asarray(image)
               
               
                current_data.append(numpyimage)
               
            cap.release()
        
            out= np.asarray(current_data)
            length = out.shape[0]
          
            # if self.clip != 'all':
            #     if out.shape[0] > self.clip:  
            #         lapse = out.shape[0] -self.clip
            #         t= torch.randint(0, lapse, (1,)).item()
            #         out=out[t:t+self.clip,...]
            if self.selection: 
                try:
                  #  selected=random.sample(range(0, out.shape[0]), self.num_frames)
                   # out =out[selected,...]
                
                   # timestep = (torch.tensor(selected) / length)[...,None]
                    start_t = random.randint(0, out.shape[0] - self.num_frames)
              #  selected=random.sample(range(0, out.shape[0]), self.num_frames)
               # out =out[selected,...]
                    out=out[ start_t:start_t + self.num_frames, :, :]
                    timestep =0
                  #  timestep = (torch.tensor(selected) )[...,None]
                    
                except:
                    out=out
                    timestep =(torch.arange(0, length)/ length )[...,None]
            else:
                timestep = 0
        
             # Clip and normalize the images
            out_clipped = np.clip(out, np.quantile(out, 0.001), np.quantile(out, 0.999))
            out_normalized = (out_clipped - np.min(out_clipped)) / (np.max(out_clipped) - np.min(out_clipped))
            out = torch.tensor(out_normalized)
           
            # if self.clip != 'all':
            #     vid = torch.zeros(self.clip,self.img_size,self.img_size)
            #     vid[:out.shape[0], ...] = out
                
            #     if self.num_frames < self.clip:
                    
            #         selected=random.sample(range(1, self.clip), self.num_frames)
            #         vid =vid[selected,...]
            #     elif self.num_frames == self.clip/2:
            #         vid = vid[ ::2, ...]
            #         print('got only every second channel')
            #     else: 
            #         vid = vid
            


            if self.img_size == 112:
                downsample = nn.AvgPool2d(kernel_size=2, stride=2)
                vid = downsample(out)
            output= {
                    "vid": vid,
                # "sites": sites,
                #  "mask": mask,
                    'time': timestep,
                    'label': label  
                }

            return  vid, label, fps, filename
          #  return vid, label
        
        print('got loaders')



class LUS_video_bedlus(torch.utils.data.Dataset):

        def __init__(self, datapath= '/gpfs/gibbs/project/hartley/jw3234/medfuncta/vidfuncta/data/BEDLUS/Tina_cropped_sorted', labelsfile = '/gpfs/gibbs/project/hartley/jw3234/medfuncta/vidfuncta/data/BEDLUS/MLSC_Lung_Annotations.CSV', img_size=112, transform=None, clip = 'all', num_frames = 4, selection = True, sobel = False):
            # /home/julia/pocovid/data/pocus_videos/convex
            self.datapath= datapath
            
            self.transform = transform
         #   self.image_paths = list(paths.list_images( datapath))


            file_paths = []
            for dirpath, dirnames, filenames in os.walk(self.datapath):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    file_paths.append(file_path)
            self.image_paths = file_paths # [os.path.join(self.datapath, f) for f in os.listdir(self.datapath)]
            print('paths', self.image_paths)
            print('len', len(self.image_paths))
            
            self.img_size = img_size
            self.clip = clip
            self.num_frames = num_frames
            self.selection = selection
            sheet_name = "ClipLevelClassificationCrowd"  # or use the sheet index like 0
          
            self.labels_df = pd.read_csv(labelsfile)


            try:
                self.labels_df.dropna(subset=['Bline_classification_by_crowd'], inplace=True)
            except:
                pass


            
        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image_names=self.image_paths[idx]
           
            filename = image_names.split('/')[-1].split('.')[0]
            
            label = self.labels_df.loc[self.labels_df['DICOMFIlename'] == filename, 'Bline_classification_by_crowd'].values
            
            curr_vid_path= os.path.join( image_names)
           
            cap = cv2.VideoCapture(curr_vid_path)
            fps = cap.get(cv2.CAP_PROP_FPS)

          
            current_data = []
            
            while cap.isOpened():
                frame_id = cap.get(1)

                ret, frame = cap.read()
                if (ret != True):
                    break
                image = cv2.resize(frame, (224,224))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
               # plt.figure()
              #  plt.imshow(image, cmap='gray')  # Tensor format: [batch, channel, height, width]
  

                numpyimage = np.asarray(image)
               
               
                current_data.append(numpyimage)
               
            cap.release()

            out= np.asarray(current_data)

            length = out.shape[0]
            if length < self.num_frames:
              print('out', out.shape)
              out_repeated = np.tile(out, (2, 1, 1))[:6]
              out = out_repeated
              print('out2', out.shape)

            # if self.clip != 'all':
            #     if out.shape[0] > self.clip:  
            #         lapse = out.shape[0] -self.clip
            #         t= torch.randint(0, lapse, (1,)).item()
            #         out=out[t:t+self.clip,...]
            if self.selection: 
                try:
                  #  selected=random.sample(range(0, out.shape[0]), self.num_frames)
                   # out =out[selected,...]
                
                #   # timestep = (torch.tensor(selected) / length)[...,None]
                    start_t = random.randint(0, out.shape[0] - self.num_frames)
                   # selected=random.sample(range(0, out.shape[0]), self.num_frames)
                   # out =out[selected,...]
                    out=out[ start_t:start_t + self.num_frames, :, :]
                    timestep =0
                  #  timestep = (torch.tensor(selected) )[...,None]
                    
                except:
                    out=out
                    timestep =(torch.arange(0, length)/ length )[...,None]
            else:
                timestep = 0
            
            try:
             # Clip and normalize the images
                out_clipped = np.clip(out, np.quantile(out, 0.001), np.quantile(out, 0.999))
                out_normalized = (out_clipped - np.min(out_clipped)) / (np.max(out_clipped) - np.min(out_clipped))
                out = torch.tensor(out_normalized)
                if self.img_size == 112:
                    downsample = nn.AvgPool2d(kernel_size=2, stride=2)
                    vid = downsample(out)
            except:
                vid=torch.nan

            output= {
                    "vid": vid,
                    'time': timestep,
                    'label': label  
                }

            return  vid, label, fps, filename
        
        print('got loaders')


class BUV_video(torch.utils.data.Dataset):

        def __init__(self, datapath= '/gpfs/gibbs/project/hartley/jw3234/medfuncta/vidfuncta/data/BUV_avis', labelsfile = None , img_size=112, transform=None, clip = 'all', num_frames = 6, selection = True):
            self.datapath= datapath
            
            self.transform = transform
            file_paths = []
            for dirpath, dirnames, filenames in os.walk(self.datapath):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    file_paths.append(file_path)
            self.image_paths = file_paths # [os.path.join(self.datapath, f) for f in os.listdir(self.datapath)]
            print('paths', self.image_paths)
            print('len', len(self.image_paths))
            self.img_size = img_size
            self.clip = clip
            self.num_frames = num_frames
            self.selection = selection


        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
           
            image_names=self.image_paths[idx]
            try:
                label = image_names.split('/')[-2]
            except:
                label = 4
          
            curr_vid_path= os.path.join( image_names)           
            cap = cv2.VideoCapture(curr_vid_path)
            fps = cap.get(cv2.CAP_PROP_FPS)

            current_data = []
            
            while cap.isOpened():
                frame_id = cap.get(1)

                ret, frame = cap.read()
                if (ret != True):
                    break
                image = cv2.resize(frame, (224,224))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                numpyimage = np.asarray(image)
                current_data.append(numpyimage)
               
            cap.release()
        
            out= np.asarray(current_data)
            if self.selection:
                start_t = random.randint(0, out.shape[0] -  self.num_frames)
                selected=random.sample(range(0, out.shape[0]), self.num_frames)
                out =out[selected,...]

             # Clip and normalize the images
            out_clipped = np.clip(out, np.quantile(out, 0.001), np.quantile(out, 0.999))
            out_normalized = (out_clipped - np.min(out_clipped)) / (np.max(out_clipped) - np.min(out_clipped))
            out = torch.tensor(out_normalized)

            if self.img_size == 112:
                downsample = nn.AvgPool2d(kernel_size=2, stride=2)
                vid = downsample(out)
            return  vid, label , fps, image_names
        
        print('got loaders')




class LUS_video_mixed(torch.utils.data.Dataset):  

        def __init__(self, datapath= '/gpfs/gibbs/project/hartley/jw3234/medfuncta/vidfuncta/data/mixedset_all', labelsfile = None, img_size=112, transform=None, clip = 'all', num_frames = 6, selection = True, sobel = False):
            # /home/julia/pocovid/data/pocus_videos/convex
            self.datapath= datapath
            
            self.transform = transform
     
            self.image_paths = []
            image_extensions = ('.mp4', '.gif', '.png', '.bmp', '.gif', '.tiff', '.webp', '.avi')
            for root, dirs, files in os.walk(self.datapath):
                for file in files:
                    if file.lower().endswith(image_extensions):
                        self.image_paths.append(os.path.join(root, file))

            print('paths', self.image_paths)
            print('len', len(self.image_paths))
            self.img_size = img_size
            self.clip = clip
            self.num_frames = num_frames
            self.sobel = sobel
            self.selection = selection

           

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
           
            image_names=self.image_paths[idx]
            filename = image_names.split('/',4)[-1]

            
                

            if 'echo' in image_names or '0X' in image_names:
                    label =0
                    
            elif 'lus' in image_names or '_' in image_names:
                    label = 1
            else:
                    label = 2   #buv
            filename
            curr_vid_path= os.path.join( image_names)
            cap = cv2.VideoCapture(curr_vid_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
        
          
            current_data = []
            
            while cap.isOpened():
                frame_id = cap.get(1)

                ret, frame = cap.read()
                if (ret != True):
                    break
                image = cv2.resize(frame, (224,224))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
               

                numpyimage = np.asarray(image)
               
               
                current_data.append(numpyimage)
               
            cap.release()
        
            out= np.asarray(current_data)
          
   
            if self.selection: 
                try:
                    selected=random.sample(range(0, out.shape[0]), self.num_frames)
                    out =out[selected,...]
                except:
                    out=out
            if out.shape[0]>60:
                out=out[:60,...]
            
        
             # Clip and normalize the images
            try:
             # Clip and normalize the images
                out_clipped = np.clip(out, np.quantile(out, 0.001), np.quantile(out, 0.999))
                out_normalized = (out_clipped - np.min(out_clipped)) / (np.max(out_clipped) - np.min(out_clipped))
                out = torch.tensor(out_normalized)
                if self.img_size == 112:
                    downsample = nn.AvgPool2d(kernel_size=2, stride=2)
                    vid = downsample(out)
            except:
                vid=torch.nan
       
            if self.sobel == True:

              s= sobel_filter(vid)  #should be of size (num_frames, 1, 112, 112)
              
              vid = torch.cat((vid.unsqueeze(1),s), dim= 1)

            return vid, label, fps, filename
        
        print('got loaders')




def get_dataset(args, only_test=False, all=False, double = False):
    train_set = None
    val_set = None
    test_set = None
    print('args', args.dataset)

    if args.dataset == 'chestmnist':
        from medmnist import ChestMNIST
        transforms = T.Compose([
            T.ToTensor(),
            T.Grayscale()
        ])

        train_set = ChestMNIST(split='train', transform=transforms, download='True', size=args.img_size)
        val_set = ChestMNIST(split='val', transform=transforms, download='True', size=args.img_size)
        test_set = ChestMNIST(split='test', transform=transforms, download='True', size=args.img_size)

        print(f'Training set containing {len(train_set)} images.')
        print(f'Test set containing {len(test_set)} images.')

        args.data_type = 'img'
        args.in_size, args.out_size = 2, 1
        args.data_size = (1, args.img_size, args.img_size)

    elif args.dataset == 'pneumoniamnist':
        from medmnist import PneumoniaMNIST
        transforms = T.Compose([
            T.ToTensor(),
            T.Grayscale()
        ])

        train_set = PneumoniaMNIST(split='train', transform=transforms, download='True', size=args.img_size)
        val_set = PneumoniaMNIST(split='val', transform=transforms, download='True', size=args.img_size)
        test_set = PneumoniaMNIST(split='test', transform=transforms, download='True', size=args.img_size)

        print(f'Training set containing {len(train_set)} images.')
        print(f'Test set containing {len(test_set)} images.')

        args.data_type = 'img'
        args.in_size, args.out_size = 2, 1
        args.data_size = (1, args.img_size, args.img_size)

    elif args.dataset == 'retinamnist':
        from medmnist import RetinaMNIST
        transforms = T.Compose([
            T.ToTensor(),
        ])
        train_set = RetinaMNIST(split='train', transform=transforms, download='True', size=args.img_size)
        val_set = RetinaMNIST(split='val', transform=transforms, download='True', size=args.img_size)
        test_set = RetinaMNIST(split='test', transform=transforms, download='True', size=args.img_size)

        print(f'Training set containing {len(train_set)} images.')
        print(f'Test set containing {len(test_set)} images.')

        args.data_type = 'img'
        args.in_size, args.out_size = 2, 3
        args.data_size = (3, args.img_size, args.img_size)

    elif args.dataset == 'dermamnist':
        from medmnist import DermaMNIST
        transforms = T.Compose([
            T.ToTensor(),
        ])
        train_set = DermaMNIST(split='train', transform=transforms, download='True', size=args.img_size)
        val_set = DermaMNIST(split='val', transform=transforms, download='True', size=args.img_size)
        test_set = DermaMNIST(split='test', transform=transforms, download='True', size=args.img_size)

        print(f'Training set containing {len(train_set)} images.')
        print(f'Test set containing {len(test_set)} images.')

        args.data_type = 'img'
        args.in_size, args.out_size = 2, 3
        args.data_size = (3, args.img_size, args.img_size)

    elif args.dataset == 'octmnist':
        from medmnist import OCTMNIST
        transforms = T.Compose([
            T.ToTensor(),
            T.Grayscale()
        ])
        train_set = OCTMNIST(split='train', transform=transforms, download='True', size=args.img_size)
        val_set = OCTMNIST(split='val', transform=transforms, download='True', size=args.img_size)
        test_set = OCTMNIST(split='test', transform=transforms, download='True', size=args.img_size)

        print(f'Training set containing {len(train_set)} images.')
        print(f'Test set containing {len(test_set)} images.')

        args.data_type = 'img'
        args.in_size, args.out_size = 2, 1
        args.data_size = (1, args.img_size, args.img_size)

    elif args.dataset == 'pathmnist':
        from medmnist import PathMNIST
        transforms = T.Compose([
            T.ToTensor(),
        ])
        train_set = PathMNIST(split='train', transform=transforms, download='True', size=args.img_size)
        val_set = PathMNIST(split='val', transform=transforms, download='True', size=args.img_size)
        test_set = PathMNIST(split='test', transform=transforms, download='True', size=args.img_size)

        print(f'Training set containing {len(train_set)} images.')
        print(f'Test set containing {len(test_set)} images.')

        args.data_type = 'img'
        args.in_size, args.out_size = 2, 3
        args.data_size = (3, args.img_size, args.img_size)

 
 
    elif args.dataset == 'mri':
        print('we got MRI dataset')
        dataset = JPGDataset(img_size=args.img_size) 
        train_set = dataset
        val_set = dataset
        test_set = dataset
      

        print(f'Training set containing {len(train_set)} images.')
        print(f'Test set containing {len(test_set)} images.')
        args.data_type = 'img'
        args.in_size, args.out_size = 2, 1
        args.data_size = (1, args.img_size, args.img_size)

    elif args.dataset == 'lus':
        print('we got LUS dataset')
        dataset = LUS_video_blue(img_size=args.img_size, clip=args.clip)
        # Define split sizes
        train_size = int(0.8 * len(dataset))  # 80% for training
        val_size = int(0.1 * len(dataset))  # 10% for validation
        test_size = len(dataset) - train_size - val_size # 10% for testing
        generator = torch.Generator().manual_seed(42)

        train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=generator)
 
        print(f'Training set containing {len(train_set)} images.')
        print(f'Test set containing {len(test_set)} images.')
        args.data_type = 'img'
        args.in_size, args.out_size = 2, 1
        args.data_size = (1, args.img_size, args.img_size)


       
    elif args.dataset == 'lusvideo' or args.dataset== 'autoregressive':
       
       # vid3d = Videoto3D(img_size=args.img_size)
       # args.videos, width=64, height=64, depth=60, framerate=1)
        if args.option =='blue':
            dataset = LUS_video_blue(img_size=args.img_size, clip=args.clip, num_frames=args.num_frames, selection=args.selection, sobel = args.sobel)
            print('got LUS video blue')
        elif args.option == 'newlus':
            dataset = LUS_video_newlus(img_size=args.img_size, clip=args.clip, num_frames=args.num_frames, selection=args.selection, sobel = args.sobel)
            print('got new LUS video ')
        elif args.option == 'mixed':
            dataset = LUS_video_mixed(img_size=args.img_size, clip=args.clip, num_frames=args.num_frames, selection=args.selection)
            print('got mixed set ')
        elif args.option =='bedlus':
            print('got bedlus dataset')
            dataset = LUS_video_bedlus(img_size=args.img_size, clip=args.clip, num_frames=args.num_frames, selection=args.selection, sobel = args.sobel)

        elif args.option == 'buv':
            dataset = BUV_video(img_size=args.img_size, clip=args.clip, num_frames=args.num_frames, selection=args.selection)
            print('got BUV set ')

        else:
            dataset = LUS_video(img_size=args.img_size, clip=args.clip)
            print('got LUS video')
   
        train_size = int(0.8 * len(dataset))  # 80% for training
        val_size = int(0.1 * len(dataset)) +1 # 10% for validation
        test_size = len(dataset) - train_size - val_size # 10% for testing
        generator = torch.Generator().manual_seed(42)
        train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=generator)
        print(f'Training set containing {len(train_set)} images.')
        print(f'Test set containing {len(test_set)} images.')

        if args.sobel == True:
            args.out_size = 2


    elif  args.dataset =='framewise':
        
            if args.option == 'blue':
                print('load LUS video blue')
                dataset = LUS_video_blue(img_size=args.img_size, clip=args.clip, num_frames=args.num_frames)

            else:
                dataset = LUS_video(img_size=args.img_size, clip=args.clip)
            train_size = int(0.8 * len(dataset))  # 80% for training
            val_size = int(0.1 * len(dataset)) +1 # 10% for validation
            test_size = len(dataset) - train_size - val_size # 10% for testing
            generator = torch.Generator().manual_seed(42)
            train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=generator)
            
            print(f'Training set containing {len(train_set)} images.')
            print(f'Test set containing {len(test_set)} images.')
            args.data_type = 'img'
            args.in_size, args.out_size = 2, 1
            args.data_size = (1, args.img_size, args.img_size)

    else:
        raise NotImplementedError()

    if only_test:
        return test_set

    elif all:
        return train_set, val_set, test_set

    else:
        return train_set, test_set
