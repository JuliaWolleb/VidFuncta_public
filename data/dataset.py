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



class JPGDataset(Dataset):
    def __init__(self, folder_path = './data/outliers', img_size = 112, transform=None):
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
            return out, label



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
                numpyimage = np.asarray(image)
                current_data.append(numpyimage)
               
            cap.release()
        
            out= np.asarray(current_data)
            length = out.shape[0]

            if self.selection: 
                try:
                    selected=random.sample(range(0, out.shape[0]), self.num_frames)
                    out =out[selected,...]  
                 #   start_t = random.randint(0, out.shape[0] - self.num_frames)
                 #   out=out[ start_t:start_t + self.num_frames, :, :]
                    timestep =0
                    
                except:
                    out=out
                    timestep =(torch.arange(0, length)/ length )[...,None]
            else:
                timestep = 0
        
             # Clip and normalize the images
            out_clipped = np.clip(out, np.quantile(out, 0.001), np.quantile(out, 0.999))
            out_normalized = (out_clipped - np.min(out_clipped)) / (np.max(out_clipped) - np.min(out_clipped))
            out = torch.tensor(out_normalized)
           

            if self.img_size == 112:
                downsample = nn.AvgPool2d(kernel_size=2, stride=2)
                vid = downsample(out)
            output= {
                    "vid": vid,
                    'time': timestep,
                    'label': label  
                }

            return  vid, label, fps, filename
        
        print('got loaders')



class LUS_video_bedlus(torch.utils.data.Dataset):

        def __init__(self, datapath= './data/lung', labelsfile = None, img_size=112, transform=None, clip = 'all', num_frames = 4, selection = True, sobel = False):
            # /home/julia/pocovid/data/pocus_videos/convex
            self.datapath= datapath
            
            self.transform = transform
         #   self.image_paths = list(paths.list_images( datapath))


            file_paths = []
            for dirpath, dirnames, filenames in os.walk(self.datapath):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    file_paths.append(file_path)
            self.image_paths = file_paths 
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
           
            filename = image_names.split('/')[-1].split('.')[0]
            label = image_names.split('/')[-2]
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

            length = out.shape[0]
            if length < self.num_frames:
              out_repeated = np.tile(out, (2, 1, 1))[:self.num_frames]
              out = out_repeated


            if self.selection: 
                try:
                    selected=random.sample(range(0, out.shape[0]), self.num_frames)
                    out =out[selected,...]             
                   # start_t = random.randint(0, out.shape[0] - self.num_frames)
                   # out=out[ start_t:start_t + self.num_frames, :, :]
                    timestep =0
                 
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

        def __init__(self, datapath= '/data/breast', labelsfile = None , img_size=112, transform=None, clip = 'all', num_frames = 6, selection = True):
            self.datapath= datapath
            
            self.transform = transform
            file_paths = []
            for dirpath, dirnames, filenames in os.walk(self.datapath):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    file_paths.append(file_path)
            self.image_paths = file_paths
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

        def __init__(self, datapath= './data/mixedset_all', labelsfile = None, img_size=112, transform=None, clip = 'all', num_frames = 6, selection = True, sobel = False):
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


            if 'cardiac' in image_names or '0X' in image_names:
                    label =0  #cardiac
                    
            elif 'lung' in image_names or '_' in image_names:
                    label = 1   #lung
            else:
                    label = 2   #breast
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
       
            return vid, label, fps, filename
        print('got loaders')




def get_dataset(args, only_test=False, all=False, double = False):
    train_set = None
    val_set = None
    test_set = None
    print('args', args.dataset)


    if args.dataset == 'outlier':
        print('we got a dataset of outliers')
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
       
        if args.option =='blue':
            dataset = LUS_video_blue(img_size=args.img_size, clip=args.clip, num_frames=args.num_frames, selection=args.selection, sobel = args.sobel)
            print('got LUS video blue')
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

    else:
        raise NotImplementedError()

    if only_test:
        return test_set

    elif all:
        return train_set, val_set, test_set

    else:
        return train_set, test_set
