import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from data_process import feature_label_DataSet
import torch.nn as nn
import queue
import tqdm
from sklearn.metrics import roc_auc_score
import sys
import os

class user_combined_features(nn.Module):
    def __init__(self, paragraph_in_dim=128, paragraph_out_dim=32):
        super().__init__()
        self.paragraph_dim = paragraph_in_dim
        self.paragraph_out_dim = paragraph_out_dim
        
        self.embedding = nn.Embedding(self.paragraph_dim, self.paragraph_out_dim)
        self.fc = nn.Linear(self.paragraph_out_dim*2, self.paragraph_out_dim)
        self.init_weights()

    def init_weights(self):
        #initrange = 1.0/16
        initrange = 1.0/11.3
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()


    def forward(self, content):
        user_id = content[:,0].to(torch.int64)
        title_vec = content[:,1:].float()
        user_feature = self.embedding(user_id)
        combined_features = torch.cat((user_feature,title_vec),1)
        result_user_feature = self.fc(combined_features)
        return result_user_feature

class query_combined_features(nn.Module):
    def __init__(self, paragraph_in_dim=128, paragraph_out_dim=32):
        super().__init__()
        self.paragraph_dim = paragraph_in_dim
        self.paragraph_out_dim = paragraph_out_dim
        
        self.embedding = nn.Embedding(self.paragraph_dim, self.paragraph_out_dim)
        self.fc = nn.Linear(self.paragraph_out_dim*2, self.paragraph_out_dim)
        self.init_weights()

    def init_weights(self):
        #initrange = 1.0/16
        initrange = 1.0/11.3
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, content):
        item_id = content[:, 0].to(torch.int64)
        title_vec = content[:, 1:]
        item_feature = self.embedding(item_id)
        item_combined_features = torch.cat((item_feature,title_vec),1)
        result_item_feature = self.fc(item_combined_features)
        return result_item_feature



class LR_net(nn.Module):
    def __init__(self, user_combined_features_model, query_combined_features_model):
        super().__init__()
        self.user_combined_features_model = user_combined_features_model
        self.query_combined_features_model = query_combined_features_model

    def forward(self, X_data):
        user_features = self.user_combined_features_model(X_data[:,:129])
        query_features = self.query_combined_features_model(X_data[:,129:])
        inner_product = torch.sum(user_features.mul(query_features),dim=1)
        result = torch.sigmoid(inner_product)
        return result

def test_auc(LR_model, val_dataloader):
    LR_model.eval()
    targets, predicts = list(), list()
    for step, sample_batched in enumerate(val_dataloader):
        batch = tuple(t.cuda() for t in sample_batched)
        feature, label = batch
        targets.extend(label.tolist())
        temp = len(set(targets))
        if temp == 1:
            return -1    
        out = LR_model(feature)
        y = out.squeeze()
        predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts)

def test_loss(LR_model, val_dataloader, criterion):
    targets, predicts = list(), list()
    loss_avg = 0
    with torch.no_grad():
        for step, sample_batched in enumerate(val_dataloader):
            batch = tuple(t.cuda() for t in sample_batched)
            X_data, Y_data = batch
            out = LR_model(X_data)
            out = out.squeeze()
            loss = criterion(out, Y_data.float())
            loss_avg += loss.mean().item()
            targets.extend(Y_data.tolist())
            y = out.squeeze()
            predicts.extend(y.tolist())
        loss_avg = loss_avg/(step+1)
        try:
            auc = roc_auc_score(targets, predicts)
        except:
           auc = -1
    return loss_avg, auc


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    #filename = "data/temp_train_data"
    #filename = "data/t"
    filename = sys.argv[1]
    print("data file is %s"%filename)
    print("model init begin")
    user_combined_features_model = user_combined_features(paragraph_in_dim=1000, paragraph_out_dim=128)
    query_combined_features_model = query_combined_features(paragraph_in_dim=10000, paragraph_out_dim=128)
    for i in range(1,20):
        user_combined_features_model.load_state_dict(torch.load("../model_train/model/user_model_state_dict_%d"%i))
        query_combined_features_model.load_state_dict(torch.load("../model_train/model/query_model_state_dict_%d"%i)) 
        LR_model = LR_net(user_combined_features_model, query_combined_features_model)
        LR_model=nn.DataParallel(LR_model).cuda() # multi-GPU

        print("feature_label_DataSet begin")
        dataset = feature_label_DataSet(filename, block_size=1024*16*16*4)
        print("feature label tensor begin")
        with torch.no_grad():
            for block_data in dataset:
                feature_tensor = torch.tensor([f[0] for f in block_data], dtype=torch.float)
                label_tensor = torch.tensor([f[1] for f in block_data], dtype=torch.int)
                dataset_tensor = TensorDataset(feature_tensor, label_tensor)
                test_dataloader = DataLoader(dataset_tensor, batch_size=1024*16*16*4, shuffle=False)
                auc = test_auc(LR_model, test_dataloader) 
                print("i=%d\tauc=%f"%(i,auc))
