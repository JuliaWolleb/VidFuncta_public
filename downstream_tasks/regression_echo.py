import os
import torch
import sys
import random
sys.path.append(".")
#sys.path.append("..")
from data.dataset import get_dataset
from common.utils import  Logger
import wandb
import time
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, Subset
import time
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models
from models.model_wrapper import ModelWrapper
import matplotlib.pyplot as plt
from common.utils import set_random_seed
from torch.utils.data import DataLoader, Dataset, random_split
#from torchmetrics.classification import BinaryF1Score, ConfusionMatrix, MulticlassPrecisionRecallCurve, ROC, AUROC, AveragePrecision
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, \
    recall_score, confusion_matrix, \
    balanced_accuracy_score, roc_curve, f1_score, average_precision_score, mean_absolute_error, root_mean_squared_error, r2_score
import numpy as np
from torch.nn import BCEWithLogitsLoss
import warnings
from pooling import Pooling, Pooling_CNN, Conv3DClassifier
from transformer import TransformerRegression, ViT, TransformerClassifier, Conv3DClassifier
from typing import Any, Dict, Optional
from lstm import LSTMRegression
import gc

gc.collect()
gc.set_threshold(0)
#import logging



class NFDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=False)
        m = data['modulations'].float()
        
      #  return m,  data['label']

        v  = data['v']
    #    
    # m = ( m -  m.min() ) / (  m.max()  -  m.min())
        m= ( m -  m.mean() ) / (  m.std())
        v= ( v -  v.mean() ) / (  v.std())
        return m, v, data['label'], data['ssim3d']


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        

    def forward(self, x, v):
      
       # x=x#+v
        return self.network(v)


def compute_metrics_echo(targets, logits):
    # Check if targets are one-hot encoded (multiclass) or not (binary)
    #print(targets, logits)
    targets = np.array(targets)
    logits = np.array(logits)#[...,None]
   
    print('length', targets.shape, logits.shape)

   # mean_absolute_error(y_true, y_pred, multioutput='raw_values')
    mae = mean_absolute_error(targets[:,0], logits[:,0])
    print('computed mae', mae)
    rmse = root_mean_squared_error(targets, logits)
    r2 = r2_score(targets, logits)
    metric_dict = {
                'mae': mae,
                'r2': r2,
                'rmse': rmse } # Calculating specificity
       

    return metric_dict


def train(model, train_loader, criterion, optimizer, device, num_classes):
    model.train()
    total_loss = 0.0
    target_l = []
    logit_l = []
  #  inr = torch.load("./nfsets/echo_framewise_addsub_2048_second/model.pt" ).to(device)
    
  #  inr = ModelWrapper(args, inr)
    for data, v, labels,_ in train_loader:
     #   
       # data, v, labels = data.to(device), v.to(device), labels.to(device)
        data, v, labels = data.to(device),  v.to(device), labels.to(device)
       #data = data[:, ::3, :]
        start = random.randint(1, 50)

        # Generate indices for every 3rd frame from start
        indices = torch.arange(start, 120, step=2)
        cut = torch.randint(1,119,(1,)) 
        data = torch.cat((data[:,cut:,...],data[:, :cut, ...]), dim=1)
        data = data[:, indices, :]
        data = data[:,:40,...]
        if random.random() < 0.5:
            data= data.flip(dims=[1]) 
       # print('v', v.shape)
       
        # inr.model.modulations = data[0,:1,...].to(device)
        # inr.model.vidm = v[0,:1,...].to(device)
        # out1 = inr()
        
        # print('out1', out1.shape, out1.max(), out1.min())
        # plt.figure(1)
        # plt.imshow(out1[0,0,...].detach().cpu().numpy())
        # plt.savefig("./img1.png")
       
      
        # wandb.log({'my_plot': plt})
       # plt.close()

       # data = torch.flatten(data, 1,2)

       # print('label', labels)
        data=data + (0.05)*torch.randn(data.shape).to(device)
        v=v + (0.05)*torch.randn(v.shape).to(device)

        
      #  labels=F.one_hot(torch.tensor(labels).long(), rint)'classes=2)
       
       # labels = labels.squeeze()
      
       
        if  data.isnan().any():
                print('got nan')
                continue
        #labels = labels.squeeze()
        optimizer.zero_grad()
                
        dropout_p = 0.1  # e.g., 20% of pixels will be zeroed

        # Apply element-wise dropout during training
        data_dropped = F.dropout(data, p=dropout_p, training=True)
        v_dropped =F.dropout(v, p=dropout_p, training=True)
        out = model(data_dropped,v_dropped)
       # data=data.view(10,60, -1)  
       # out = model(data)
     #   print('shapes', out.shape, labels.shape)

        loss = criterion(out, labels.float())
       # out = torch.flatten(out)

        
     #   out = torch.flatten(out)

        target_l.extend([*(labels.cpu().numpy()).astype('float')])

        logit_l.extend([*out.detach().cpu().numpy()])
        if  out.isnan().any():
                print('got nan')
                continue
        
       
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    train_epoch_metrics = compute_metrics_echo(target_l, logit_l)
    print('train metrics', train_epoch_metrics)

    return total_loss / len(train_loader), train_epoch_metrics


def evaluate(model, val_loader, criterion, device, num_classes):
    #model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    target_l = []
    logit_l = []

    with torch.no_grad():
        ssimlist = []
        losslist = []
        for data, v, labels, ssim3d in val_loader:
            # data, v, labels = data.to(device), v.to(device), labels.to(device)
            data,v,labels = data.to(device),  v.to(device), labels.to(device)
            data = data[:, ::2, :]
            data = data[:,:40,...]




           # labels = labels.squeeze()
         #   data=data + (0.001)*torch.t(data.shape).to(device)

         
          #  labels=F.one_hot(torch.tensor(labels).long(), num_classes=2)
            
          #  data=data.view(10,60, -1)  
            if  data.isnan().any():
                print('got nan')
                continue
            out = model(data,v)
        #    out = model(data)
           
          #  print('shape', out.shape, labels.shape)
            criterion2 = nn.L1Loss(reduction='none')
            loss = criterion2(out, labels.float())
            loss_values = loss
            ssimlist.extend(ssim3d.cpu().numpy())
            losslist.extend(loss_values.cpu().numpy())

           # out = torch.flatten(out)
           
          #  mae = mean_absolute_error(labels[:,0].cpu().numpy(), out[:,0].cpu().numpy())
      
            target_l.extend([*(labels.cpu().numpy()).astype('float')])

            logit_l.extend([*out.detach().cpu().numpy()])
          
            

            total_loss += loss.mean().item()

         
            total += labels.size(0)

    val_epoch_metrics = compute_metrics_echo(target_l, logit_l)

    # Scatter plot: label vs loss
    # plt.figure(figsize=(8, 6))
    # plt.scatter(ssimlist, losslist, alpha=0.5)
    # plt.xlabel("SSIM3d")
    # plt.ylabel("Mean Average Error")
    # plt.title("Scatterplot of SSIM3d vs MAE")
    # plt.grid(True)

    # # Save the plot as a PNG file
  #  plt.savefig("ssim3d_vs_MAE.png", dpi=300)

   # plt.show()

    return total_loss / len(val_loader),  val_epoch_metrics['mae'], val_epoch_metrics['rmse'], val_epoch_metrics['r2'], val_epoch_metrics



class Args:
    def __init__(self):
        self.data_dir = "./reconstructions/echo_separate_512/nfset" 
        self.classifier ='transformer'  # simple or resnet or efficientnet
        self.mode = 'grayscale'  # grayscale or tgb
        self.input_dim = 512  #8192 2048 
        self.num_classes = 1
        self.learning_rate = 1e-4
        self.num_epochs = 60
        self.seed = 42
        self.comment = 'plain'
        self.img_size=  112
        self.batch_size = 4
        self.num_frames = 120

def prefix_dict(d: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in d.items()}


def main(args):
    # Define device
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    wandb.init(project="echo")
    run_name = f"model={args.classifier}_inputsize={args.input_dim}_lr={args.learning_rate}_dataset={args.data_dir}_clip={args.num_frames}_comment={args.comment}"
        
    wandb.run.name = run_name
    """ Enable determinism """
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """ Define Dataset and Dataloader """
    if args.classifier == 'simple' or 'pooling' or 'transformer' or 'lstm' or 'poolingcnn' or 'conv3d':
        train_set = NFDataset(os.path.join(args.data_dir, "train"))
#        val_set = NFDataset(os.path.join(args.data_dir, "val"))
        test_set = NFDataset(os.path.join(args.data_dir, "test"))
        vset = torch.utils.data.ConcatDataset([train_set,  test_set])
        k_folds = 5
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=1)
        
        trainval_size = int(0.9* len( vset))
        test_size = len( vset) - trainval_size 

        #Split the dataset
        trainval_set,  test_set = random_split(vset, [trainval_size, test_size])
   
        test_scores_mae = []
        test_scores_rmse =[]
        test_scores_r2 = []
       
        print('got nf dataset')
        for fold, (train_idx, val_idx) in enumerate(kf.split(trainval_set)):
          print(f"Fold {fold + 1}")
          torch.cuda.synchronize()
         


          if fold <6:
            train_subset = Subset(trainval_set, train_idx)
            val_subset = Subset(trainval_set, val_idx)
            train_loader = DataLoader(train_subset, batch_size=30, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_subset, batch_size=30, shuffle=False, drop_last=True)

            test_loader = DataLoader(test_set,  batch_size=1,num_workers=0, shuffle=False, drop_last=True)
            print('loaders', len(train_loader), len(val_loader))


            """ Select classification model """
            if args.classifier == 'simple':
                model = SimpleClassifier(2048, args.num_classes).to(device)
            elif args.classifier == 'pooling':
                model = Pooling(features= args.input_dim, pooling_type="attention", with_mlp=True).to(device)
                print('got pooling')
            elif args.classifier == 'poolingcnn':
                model = Pooling_CNN(features= args.input_dim, pooling_type="attention", with_cnn=True).to(device)
                print('got pooling')
            elif args.classifier == 'resnet':
                model = ResNet50Classifier(args.num_classes, mode=args.mode).to(device)
            elif args.classifier == 'efficientnet':
                model = EfficientNetB0Classifier(args.num_classes, mode=args.mode).to(device)
            elif args.classifier == 'transformer':
                model = ViT(img_size=args.input_dim).to(device)
                print('got transformer')
            elif args.classifier == 'lstm':
                model = LSTMRegression().to(device)
            elif args.classifier == 'conv3d':
                model = Conv3DClassifier().to(device)
            pytorch_total_params = sum(p.numel() for p in model.parameters())
            print(f"Parameters: {pytorch_total_params}")
            wandb.watch(model)
            """ Define optimization criterion and optimizer """
        # criterion = nn.CrossEntropyLoss()
        # criterion = BCEWithLogitsLoss()
            criterion = nn.MSELoss()
            criterion.to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

            """ Run training and validation loop """
            best_mae = 0.0
            start_time = time.time()
            for epoch in range(args.num_epochs):
                train_loss, train_epoch_metrics = train(model, train_loader, criterion, optimizer, device, args.num_classes)
                wandb.log(prefix_dict(train_epoch_metrics, "train/"), step=epoch)

                val_loss, val_mae, val_rmse, val_r2, val_epoch_metrics = evaluate(model, val_loader, criterion, device, args.num_classes)
                wandb.log(prefix_dict(val_epoch_metrics, "valid/"), step=epoch)
                if val_mae > best_mae:
                    best_mae = val_mae
                    best_model = model.state_dict()

                print(f"Epoch {epoch + 1}/{args.num_epochs}: ",
                    f"Train Loss: {train_loss:.4f} , train Acc: {train_epoch_metrics['mae']:.2f}",
                    f"| Val Loss: {val_loss:.4f}, Val mae: {val_mae:.2f}%, Val RMSE {val_rmse:.4f}%, Val r2: {val_r2:.4f}")

            
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time} seconds")

            """ Final evaluation on test set """
            model.load_state_dict(best_model)
            test_loss, test_mae, test_rmse, test_r2, test_epoch_metrics = evaluate(model, test_loader, criterion, device, args.num_classes)
            print(f"Test Loss: {test_loss:.4f}, Test mae: {test_mae:.2f}%, Test rmse: {test_rmse:.4f}%, test r2: {test_r2:.4f}")
            wandb.log(prefix_dict(test_epoch_metrics, "test/"), step=epoch)
            test_scores_mae.append(test_epoch_metrics['mae'])
            test_scores_rmse.append(test_epoch_metrics['rmse'])
            test_scores_r2.append(test_epoch_metrics['r2'])
    print('test mae', test_scores_mae)
    print('test rmse', test_scores_rmse)

    print('test r2', test_scores_r2)

    mean_mae = np.mean(test_scores_mae)
    mean_rmse = np.mean(test_scores_rmse)
    mean_r2 = np.mean(test_scores_r2)

    std_mae = np.std(test_scores_mae)
    std_rmse = np.std(test_scores_rmse)

    std_r2 = np.std(test_scores_r2)


    print(f"\nMean Test MAE: {mean_mae:.4f}")
    print(f"\nMean auroc RMSE: {mean_rmse:.4f}")
    print(f"\nMean R2: {mean_r2:.4f}")
    print(f"Std Test mae: {std_mae:.4f}")
    print(f"Std Test rmse: {std_rmse:.4f}")
    print(f"Std Test r2: {std_r2:.4f}")

if __name__ == "__main__":
    args = Args()
    main(args)