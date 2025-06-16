import os
import torch
import sys
import random
sys.path.append(".")
#sys.path.append("..")
from data.dataset import get_dataset
import time
#import wandb
from sklearn.model_selection import KFold
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models
import common.utils as utils
from common.utils import set_random_seed, prefix_dict
from torch.utils.data import DataLoader, Dataset, Subset
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
        
        label_dict = {
            'benign': 0,
            'malignant': 1,
        }
        labels = label_dict[data['label'][0]]
       
      

     #  m = ( m -  m.min() ) / (  m.max()  -  m.min())
        m= ( m -  m.mean() ) / (  m.std())
        v= ( v -  v.mean() ) / (  v.std())
     #   v=0
        M=torch.zeros((180,512))
        t=m.shape[0]
        if t >= 180:
           M=m[:180,...]
        else:
            # Repeat enough times and then truncate
            repeat_factor = (180 + t - 1) // t  # Ceiling division
            repeated = m.repeat((repeat_factor, 1))
            M= repeated[:180]

        return M, v, labels

  
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512 , 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, v):
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
        #preds = (logits > 0).astype(float)
        probabilities = 1 / (1 + np.exp(-logits))  #sigmoid
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
    for data,v, labels in train_loader:
     
       
        data, v, labels = data.to(device), v.to(device), labels.to(device)
        start = random.randint(1, 50)
        if  data.isnan().any():
                print('got nan')
                continue
        # Generate indices for every 3rd frame from start
        indices = torch.arange(start, 180, step=3)
        cut = torch.randint(1,179,(1,)) 
        data = torch.cat((data[:,cut:,...],data[:, :cut, ...]), dim=1)
        data = data[:, indices, :]
      #  labels = labels.squeeze()
     
       # data=data+ (0.05)*torch.randn(data.shape).to(device)
       # labels=F.one_hot(labels.clone().detach().long(), num_classes=2).clone().detach()


       
        optimizer.zero_grad()

      #  out=model(v)
        out = model(data,v)
       # out = model(data)
        out = torch.flatten(out)    
        labels = torch.flatten(labels) 
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
   # model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    target_l = []
    logit_l = []
   
    
    with torch.no_grad():
        for data, v,  labels in val_loader:
           
            data,v, labels = data.to(device),  v.to(device), labels.to(device)
          #  indices = torch.arange(0, 120, step=2)
           # data = data[:, indices, :]
            start =10
            indices = torch.arange(start, 180, step=3)
        
       
            data = data[:, indices, :].to(device)

           
            L= labels.clone()
         
         #  data=data+ (0.005)*torch.randn(data.shape).to(device)
          #  v=v+ (0.005)*torch.randn(v.shape).to(device)
          #  labels=F.one_hot(torch.tensor(labels.clone().detach()).long(), num_classes=2) 
            out = model(data, v)
         #   out = model(v)
         #   out = model(data)
            out = torch.flatten(out)    
            labels = torch.flatten(labels) 
          
          
            loss = criterion(out, labels.float())

            total_loss += loss.item()
            predicted= (out> 0).float()
          
           # _, predicted = torch.max(out, 1)
            
     
            target_l.extend([*(labels.cpu().numpy()).astype('float')])

            logit_l.extend([*out.detach().cpu().numpy()])
         
            
            correct += (predicted == L).sum().item()
            total += labels.size(0)

           
    val_epoch_metrics = compute_metrics(target_l, logit_l)
    
    accuracy = 100 * correct / total


    return total_loss / len(val_loader), accuracy, val_epoch_metrics['f1_score'],val_epoch_metrics['specificity'] , val_epoch_metrics['roc_auc'], val_epoch_metrics


class Args:
    def __init__(self):
        self.data_dir = "./reconstructions/buv_ours_allvideos/nfset" 
        self.classifier ='transformer'  # simple or resnet or efficientnet or pooling
        self.mode = 'grayscale'  # grayscale or tgb
        self.input_dim =  512 #8192 2048 
        self.num_classes = 1
        self.learning_rate = 1e-4
        self.num_epochs =20
        self.seed = 42
        self.dataset = 'lusvideo'
        self.img_size=  128
        self.batch_size = 10
        self.num_frames = 60
        self.clip = 60


def main(args):
    # Define device
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    #wandb.init(project="newlus")
    #wandb.run.name = 'medfuncta framewise 2048 clip 60'

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
        vset = torch.utils.data.ConcatDataset([train_set, val_set])
        k_folds = 5
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=1)
        
     #   vloader = DataLoader(vset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True,
      #                         drop_last=True)
      #  train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
      #  val_loader = DataLoader(val_set,  batch_size=1, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_set,  batch_size=10, shuffle=True, drop_last=True)
        


   
    test_scores_acc = []
    test_scores_f1 =[]
    test_scores_auroc = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(vset)):
        print(f"Fold {fold + 1}")
        train_subset = Subset(vset, train_idx)
        val_subset = Subset(vset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=10, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=10, shuffle=False)
        print('loaders', len(train_loader), len(val_loader), len(test_loader))

        """ Select classification model """
        if args.classifier == 'simple':
            model = SimpleClassifier(2048, args.num_classes).to(device)
        elif args.classifier == 'pooling':
            model = Pooling(features= args.input_dim, pooling_type="attention", with_mlp=True).to(device)
        
        elif args.classifier == 'resnet':
            model = ResNet50Classifier(args.num_classes, mode=args.mode).to(device)
        elif args.classifier == 'efficientnet':
            model = EfficientNetB0Classifier(args.num_classes, mode=args.mode).to(device)
        elif args.classifier == 'lstm':
            input_size = args.input_dim
            hidden_size = 64
            num_layers = 2
            output_size = args.num_classes
            model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size).to(device)
            
        elif args.classifier == 'transformer':
            model = ViT(args.input_dim).to(device)
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {pytorch_total_params}")

        """ Define optimization criterion and optimizer """
    #  criterion = nn.CrossEntropyLoss()
        criterion = nn.BCEWithLogitsLoss()

        
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        """ Run training and validation loop """
        best_val_roc = 0.0
        start_time = time.time()
        for epoch in range(args.num_epochs):
            train_loss, train_epoch_metrics = train(model, train_loader, criterion, optimizer, device, args.num_classes,  args.clip)
          #  wandb.log(prefix_dict(train_epoch_metrics, "train/"), step=epoch)

            val_loss, val_acc, val_f1, val_auprc, val_roc, val_epoch_metrics = evaluate(model, val_loader, criterion, device, args.num_classes, args.clip)
            val_acc2 = val_epoch_metrics['accuracy']
            print('validation', val_epoch_metrics)
           # wandb.log(prefix_dict(val_epoch_metrics, "valid/"), step=epoch)
            if val_roc > best_val_roc:
                best_val_roc = val_roc
                print('best val roc', best_val_roc)
                best_model = model.state_dict()

                test_loss, test_acc, test_f1, test_auprc, test_roc, test_epoch_metrics = evaluate(model, test_loader, criterion, device, args.num_classes, args.clip)
                print('test', test_epoch_metrics)
               # wandb.log(prefix_dict(test_epoch_metrics, "test/"), step=epoch)

            
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
       # wandb.log(prefix_dict(test_epoch_metrics, "test/"), step=epoch)
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
    print(f"\nMean auroc Accuracy: {mean_auroc:.4f}")
    print(f"\nMean f1 Accuracy: {mean_f1:.4f}")
    print(f"Std Test Accuracy: {std_acc:.4f}")
    print(f"Std Test auroc: {std_auroc:.4f}")
    print(f"Std Test fq: {std_f1:.4f}")


if __name__ == "__main__":
    args = Args()
    #wandb.login()
    main(args)
