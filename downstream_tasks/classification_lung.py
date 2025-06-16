import os
import torch
import sys
import pandas as pd
import random
sys.path.append(".")
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, Subset

#sys.path.append("..")
from data.dataset import get_dataset
import time
import wandb
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models
import common.utils as utils
from common.utils import set_random_seed, prefix_dict
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import BinaryF1Score, ConfusionMatrix, MulticlassPrecisionRecallCurve, ROC, AUROC, AveragePrecision
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, \
    recall_score, confusion_matrix, \
    balanced_accuracy_score, roc_curve, f1_score, average_precision_score
from torch.nn import BCEWithLogitsLoss
import warnings
from pooling import Pooling
from transformer import ViT

class NFDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=False)
        m = data['modulations'].float()
        v = data['v']
        m= ( m -  m.mean() ) / (  m.std())
        v= ( v -  v.mean() ) / (  v.std())
        label_dict = {
            'bline': 1,
            'nobline': 0,
        }
        bline_score  = label_dict[data['label'][0]]


        t=m.shape[0]
        if t > 120:
            labse = m.shape[0] -120
            t= torch.randint(0, labse, (1,)).item()
            M=m[t:t+120,...]
           
        else:
            # Repeat enough times and then truncate
            repeat_factor = (120 + t - 1) // t  # Ceiling division
            repeated = m.repeat((repeat_factor, 1,1,1))
            M= repeated[:120]

        return M, v, bline_score

  
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512 , 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
    def forward(self, x,v):
        return self.network(v)



def compute_metrics(targets, logits):
    # Check if targets are one-hot encoded (multiclass) or not (binary)
    
    targets = np.array(targets)
    logits = np.array(logits)
    exp_logits = np.exp(logits)
   
    default_metric_dict = {
        'accuracy': 0.0,
        'sensitivity': 0.0,
        'specificity': 0.0,
        'balanced_accuracy': 0.0,
        'roc_auc': 0.0,
        'f1_score': 0.0,
        'precision': 0.0,
        'confusion_matrix': None
    }

    if targets.ndim == 1 or targets.shape[1] == 1:
        # Binary classification
        print('got binary classification')
        probabilities = 1 / (1 + np.exp(-logits))  
        preds = (probabilities > 0.5).astype(int)
        try:
            tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()
            fpr, tpr, _ = roc_curve(targets, probabilities)
            metric_dict = {
                'accuracy': accuracy_score(targets, preds),
                'sensitivity': recall_score(targets, preds, pos_label=1),
                'specificity': recall_score(targets, 1 - preds, pos_label=1),  # Calculating specificity
                'balanced_accuracy': balanced_accuracy_score(targets, preds),
                'roc_auc': roc_auc_score(targets, probabilities),
                'auprc': average_precision_score(targets, probabilities),
                'f1_score': f1_score(targets, preds),
                'confusion_matrix': None,
                'tn': tn,
                'fp': fp,
                'fn': fn,
                'tp': tp
            }

        except ValueError as e:
            print(f"Error in metric computation: {e}")
        
            metric_dict = {
                'accuracy': accuracy_score(targets, preds),
                'sensitivity': recall_score(targets, preds, pos_label=1),
                'specificity': recall_score(targets, 1 - preds, pos_label=1),  # Calculating specificity
                'balanced_accuracy': balanced_accuracy_score(targets, preds),
                'roc_auc': roc_auc_score(targets, probabilities),
                'f1_score': f1_score(targets, preds),
                'confusion_matrix': None,
            }


    return metric_dict


def train(model, train_loader, criterion, optimizer, device, num_classes, clip):
    model.train()
    total_loss = 0.0
    target_l = []
    logit_l = []
    for data, v, labels in train_loader:

        
        data, v, labels = data.to(device), v.to(device), labels.to(device)
       
        cut = torch.randint(1,119,(1,)) 
        data = torch.cat((data[:,cut:,...],data[:, :cut, ...]), dim=1)
        if torch.rand(1).item() < 0.5:
            data = data.flip(dims=[1]) 
             
        data=data+ (0.1)*torch.randn(data.shape).to(device)
        v=v+ (0.1)*torch.randn(v.shape).to(device)

        if  data.isnan().any():
                print('got nan')
                continue
       
        optimizer.zero_grad()
        out = model(data,v)

       
        loss = criterion(out, labels.float())
        loss.backward()
        optimizer.step()

        target_l.extend([*(labels.cpu().numpy()).astype('float')])

        logit_l.extend([*out.detach().cpu().numpy()])
        
       
      

        total_loss += loss.item()
    train_epoch_metrics = compute_metrics(target_l, logit_l)
    print('train epoch', train_epoch_metrics)

    return total_loss / len(train_loader), train_epoch_metrics


def evaluate(model, val_loader, criterion, device, num_classes, clip):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    target_l = []
    logit_l = []
   
    
    with torch.no_grad():
        for data, v, labels in val_loader:
            data,v, labels = data.to(device),  v.to(device), labels.to(device)
            
            L= labels.clone()

            out = model(data, v)
            out = torch.flatten(out)    
            labels = torch.flatten(labels) 
     
            target_l.extend([*(labels.cpu().numpy()).astype('float')])

            logit_l.extend([*out.detach().cpu().numpy()])
          
            loss = criterion(out, labels.float())

            total_loss += loss.item()
            predicted= (out> 0).float()

            
            correct += (predicted == L).sum().item()
            total += labels.size(0)

           
    val_epoch_metrics = compute_metrics(target_l, logit_l)
    accuracy = 100 * correct / total


    return total_loss / len(val_loader), accuracy, val_epoch_metrics['f1_score'],val_epoch_metrics['specificity'] , val_epoch_metrics['roc_auc'], val_epoch_metrics


class Args:
    def __init__(self):
        self.data_dir = "./reconstructions/lung_ours/nfset" 
        self.classifier ='transformer'  # simple or transformer
        self.mode = 'grayscale'  
        self.input_dim =  512
        self.v_dim = 2048
        self.num_classes = 1
        self.learning_rate = 1e-4
        self.num_epochs =20
        self.seed = 42
        self.dataset = 'echo'
        self.img_size=  112


def main(args):
    # Define device
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    wandb.init(project="newlus")
    wandb.run.name = 'medfuncta framewise 2048 clip 60'

    """ Enable determinism """
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """ Define Dataset and Dataloader """
    if args.classifier == 'simple' or args.classifier == 'lstm' or args.classifier =='pooling'or args.classifier =='transformer':
        train_set = NFDataset(os.path.join(args.data_dir, "train"))
        val_set = NFDataset(os.path.join(args.data_dir, "test"))
        test_set = NFDataset(os.path.join(args.data_dir, "val"))
        print('got nf dataset')
        vset = torch.utils.data.ConcatDataset([val_set,  train_set])
        k_folds = 5
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=1)
        test_loader = DataLoader(test_set,  batch_size=10, shuffle=True, drop_last=True)

   
    else:
        raise NotImplementedError()

    test_scores_acc = []
    test_scores_f1 =[]
    test_scores_auroc = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(vset)):
        print(f"Fold {fold + 1}")
        train_subset = Subset(vset, train_idx)
        val_subset = Subset(vset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=20, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=10, shuffle=False)
        print('loaders', len(train_loader), len(val_loader), len(test_loader))

        """ Select classification model """
        if args.classifier == 'simple':
            model = SimpleClassifier(args.v_dim, args.num_classes).to(device)
  
        elif args.classifier == 'transformer':
            model = ViT( n_classes=1).to(device)

        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {pytorch_total_params}")

        """ Define optimization criterion and optimizer """
        criterion = nn.BCEWithLogitsLoss()

        criterion.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        """ Run training and validation loop """
        best_val_roc = 0.0
        start_time = time.time()
        for epoch in range(args.num_epochs):
            train_loss, train_epoch_metrics = train(model, train_loader, criterion, optimizer, device, args.num_classes,  args.clip)
            wandb.log(prefix_dict(train_epoch_metrics, "train/"), step=epoch)

            val_loss, val_acc, val_f1, val_auprc, val_roc, val_epoch_metrics = evaluate(model, val_loader, criterion, device, args.num_classes, args.clip)
            val_acc2 = val_epoch_metrics['accuracy']
            wandb.log(prefix_dict(val_epoch_metrics, "valid/"), step=epoch)
            if val_roc > best_val_roc:
                best_val_roc = val_roc
                best_model = model.state_dict()
            
            print(f"Epoch {epoch + 1}/{args.num_epochs}: ",
                f"Train Loss: {train_loss:.4f} , train Acc: {train_epoch_metrics['accuracy']:.2f}",
                f"| Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Acc 2: {val_acc2:.2f}%, Val F1: {val_f1:.4f}%, Val Prec: {val_auprc:.4f}%, Val ROC: {val_roc:.4f}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")

    
        """ Final evaluation on test set """
        model.load_state_dict(best_model)
        test_loss, test_acc, test_f1,  test_auprc, test_roc, test_epoch_metrics = evaluate(model, test_loader, criterion, device, args.num_classes, args.clip)
        test_bacc= test_epoch_metrics['balanced_accuracy'] 
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%, Test F1 Score: {test_f1:.4f}%, test Prec: {test_auprc:.4f}%, test ROC: {test_roc:.4f}")
        test_scores_acc.append(test_bacc)
        test_scores_auroc.append(test_roc)
        test_scores_f1.append(test_f1)

    print('test acc', test_scores_acc)
    print('test auroc', test_scores_auroc)
    print('test f1', test_scores_f1)
    
    mean_acc = np.mean(test_scores_acc)
    mean_auroc = np.mean(test_scores_auroc)
    mean_f1 = np.mean(test_scores_f1)

    std_acc = np.std(test_scores_acc)
    std_auroc = np.std(test_scores_auroc)

    std_f1 = np.std(test_scores_f1)


    print(f"\nMean Test Accuracy: {mean_acc:.4f}")
    print(f"Std Test Accuracy: {std_acc:.4f}")

    print(f"\nMean  Test auroc: {mean_auroc:.4f}")
    print(f"\nstd  Test auroc: {std_auroc:.4f}")

    print(f"\nMean Test  f1 : {mean_f1:.4f}")
    print(f"Std Test f1: {std_f1:.4f}")


if __name__ == "__main__":
    args = Args()
    wandb.login()
    main(args)
