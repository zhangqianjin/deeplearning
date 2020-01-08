import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from data_process import feature_label_DataSet


class user_combined_features(nn.Module):
    def __init__(self, user_role_dim_info, paragraph_in_dim=256, paragraph_out_dim=32):
        super().__init__()
        self.user_role_vocab_size = user_role_dim_info[0]
        self.user_role_embed_dim = user_role_dim_info[1]
        self.paragraph_dim = paragraph_in_dim
        self.paragraph_out_dim = paragraph_out_dim

        self.user_role_embedding = nn.Embedding(self.user_role_vocab_size, self.user_role_embed_dim)
        self.user_role_fc = nn.Linear(16, 16)
        self.user_query_title_fc = nn.Linear(self.paragraph_dim, self.paragraph_out_dim)
        self.user_query_fc = nn.Linear(self.paragraph_dim, self.paragraph_out_dim)
        self.user_hotword_fc = nn.Linear(self.paragraph_dim, self.paragraph_out_dim)
        self.user_hotword_title_fc = nn.Linear(self.paragraph_dim, self.paragraph_out_dim)
        self.fc = nn.Linear(144, 256)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.user_role_embedding.weight.data.uniform_(-initrange, initrange)
        self.user_role_fc.weight.data.uniform_(-initrange, initrange)
        self.user_role_fc.bias.data.zero_()

        self.user_query_title_fc.weight.data.uniform_(-initrange, initrange)
        self.user_query_title_fc.bias.data.zero_()

        self.user_query_fc.weight.data.uniform_(-initrange, initrange)
        self.user_query_fc.bias.data.zero_()

        self.user_hotword_fc.weight.data.uniform_(-initrange, initrange)
        self.user_hotword_fc.bias.data.zero_()

        self.user_hotword_title_fc.weight.data.uniform_(-initrange, initrange)
        self.user_hotword_title_fc.bias.data.zero_()

        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, user_content_info):
        user_role = user_content_info[:, 0]
        user_query_title = user_content_info[:, 1:257]
        user_query = user_content_info[:, 257:513]
        user_hotword = user_content_info[:, 513:769]
        user_hotword_title = user_content_info[:, 769:]


        user_role_embedded = self.user_role_embedding(user_role.long())
        user_role_vec = self.user_role_fc(user_role_embedded)
        user_query_title_vec = self.user_query_title_fc(user_query_title)
        user_query_vec = self.user_query_fc(user_query)
        user_hotword_vec = self.user_hotword_fc(user_hotword)
        user_hotword_title_vec = self.user_hotword_title_fc(user_hotword_title)


        user_combined_features = torch.cat((user_role_vec,user_query_title_vec,user_query_vec, user_hotword_vec, user_hotword_title_vec),1)
        result = self.fc(user_combined_features)
        return result


class query_combined_features(nn.Module):
    def __init__(self, query_hotrate_dim_info, paragraph_in_dim=256, paragraph_out_dim=32):
        super().__init__()
        self.query_hotrate_vocab_size = query_hotrate_dim_info[0]
        self.query_hotrate_embed_dim = query_hotrate_dim_info[1]
        self.paragraph_dim = paragraph_in_dim
        self.paragraph_out_dim = paragraph_out_dim

        self.query_hotrate_embedding = nn.Embedding(self.query_hotrate_vocab_size, self.query_hotrate_embed_dim)
        self.query_hotrate_fc = nn.Linear(16, 16)
        self.query_fc = nn.Linear(self.paragraph_dim, self.paragraph_out_dim)
        self.query_title_fc = nn.Linear(self.paragraph_dim, self.paragraph_out_dim)

        self.fc = nn.Linear(80, 256)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.query_hotrate_embedding.weight.data.uniform_(-initrange, initrange)

        self.query_fc .weight.data.uniform_(-initrange, initrange)
        self.query_fc .bias.data.zero_()

        self.query_title_fc.weight.data.uniform_(-initrange, initrange)
        self.query_title_fc.bias.data.zero_()

        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, query_content_info):
        print(query_content_info.shape)
        query_hotrate = query_content_info[:, 0]
        query_hotrate = query_hotrate.reshape(query_hotrate.shape[0], 1)
        query = query_content_info[:, 1:257]
        query_title = query_content_info[:, 257:]
        x = self.query_hotrate_embedding(torch.LongTensor([0])).repeat(query_hotrate.shape[0],1)
        # print(x)
        # print(query_hotrate.shape)
        query_hotrate_embedded = self.query_hotrate_embedding(torch.LongTensor([0])).repeat(query_hotrate.shape[0],1)*(query_hotrate)
        print(query_hotrate_embedded.shape)
        query_hotrate_vec = self.query_hotrate_fc(query_hotrate_embedded)
        query_vec = self.query_fc(query)
        print("query_vec:", query_title.shape)
        query_title_vec = self.query_title_fc(query_title)

        query_combined_features = torch.cat((query_hotrate_vec, query_vec, query_title_vec), 1)
        result = self.fc(query_combined_features)
        return result


class LR_net(nn.Module):
    def __init__(self, user_combined_features_model, query_combined_features_model):
        super().__init__()
        self.user_combined_features_model = user_combined_features_model
        self.query_combined_features_model = query_combined_features_model

    def forward(self, X_data):
        user_features = self.user_combined_features_model(X_data[:,:1025])
        print("user_features:", user_features.shape)
        query_features = self.query_combined_features_model(X_data[:,1025:1539])
        print("query_features:",query_features.shape)
        inner_product = torch.sum(user_features.mul(query_features),dim=1)
        print("inner_product.shape:",inner_product.shape)
        # result = torch.sigmoid(inner_product)
        return inner_product


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    filename = "test_data"
    dataset = feature_label_DataSet(filename)
    train_num = int(len(dataset) * 0.8)
    test_num = len(dataset) - train_num
    feature_tensor = torch.tensor([f["feature"] for f in dataset], dtype=torch.float)
    label_tensor = torch.tensor([f["label"] for f in dataset], dtype=torch.int)
    dataset_tensor = TensorDataset(feature_tensor, label_tensor)

    train_set, test_set = random_split(dataset_tensor, (train_num, test_num))
    train_dataloader = DataLoader(train_set, batch_size=80, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=4, shuffle=True)
    user_combined_features_model = user_combined_features([3, 16])
    query_combined_features_model = query_combined_features([1, 16])
    LR_model = LR_net(user_combined_features_model, query_combined_features_model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD([{'params': user_combined_features_model.parameters()},
                                 {'params': query_combined_features_model.parameters(), 'lr': 1e-4}], lr=0.01,
                                momentum=0.9)

    # print(query_combined_features_model)
    best_auc = 0

    for i_batch, sample_batched in enumerate(train_dataloader):
        batch = tuple(t for t in sample_batched)
        # batch = tuple(t.cuda() for t in sample_batched)
        feature, label = batch
        X_data = feature
        Y_data = label
        print(feature.shape)
        out = LR_model(X_data)
        out = out.squeeze()
        loss = criterion(out, Y_data.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("epoch=%d\tloss=%f" % (i_batch, loss.item()))
    for name, parameters in user_combined_features_model.named_parameters():
        print(name, ':', parameters)
    #模型保持
    torch.save(query_combined_features_model.state_dict(), "mov_combined_features_model_state_dict")
    torch.save(user_combined_features_model.state_dict(), "user_combined_features_model_state_dict")
    
    user_model = user_combined_features([3, 16])
    mov_model = query_combined_features([1, 16])
    #模型加载
    mov_model.load_state_dict(torch.load("mov_combined_features_model_state_dict"))
    user_model.load_state_dict(torch.load("user_combined_features_model_state_dict"))
    user_model.eval()
    mov_model.eval()
