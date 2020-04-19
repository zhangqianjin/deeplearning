import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from predict_data_process import feature_label_DataSet
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

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    #filename = "data/temp_train_data"
    #filename = "data/t"
    filename = sys.argv[1]
    print("data file is %s"%filename)
    print("model init begin")
    user_combined_features_model = user_combined_features(paragraph_in_dim=1000, paragraph_out_dim=128)
    user_combined_features_model.load_state_dict(torch.load("../model/user_model_state_dict_10", map_location=torch.device('cpu')))
    print("feature_label_DataSet begin")
    dataset = feature_label_DataSet(filename, block_size=1024*16)
    print("feature label tensor begin")
    with torch.no_grad():
        for block_data in dataset:
            feature_tensor = torch.tensor([f[0] for f in block_data], dtype=torch.float)
            label_tensor = torch.tensor([f[1] for f in block_data], dtype=torch.int)
            dataset_tensor = TensorDataset(feature_tensor, label_tensor)
            test_dataloader = DataLoader(dataset_tensor, batch_size=1024*16, shuffle=False)
            for step, sample_batched in enumerate(test_dataloader):
                batch = tuple(t for t in sample_batched)
                X_data, label = batch
                user_features = user_combined_features_model(X_data)
                user_features_list = user_features.tolist()
                for feature in user_features_list:
                    feature_str = ",".join(list(map(str, feature)))
                    print(feature_str)
