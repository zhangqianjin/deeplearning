import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from data_process import feature_label_DataSet
import torch.nn as nn
import matplotlib.pyplot as plt

import tqdm
from sklearn.metrics import roc_auc_score,roc_curve, precision_recall_curve


class user_combined_features(nn.Module):
    def __init__(self, paragraph_in_dim=256, paragraph_out_dim=32):
        super().__init__()
        self.paragraph_dim = paragraph_in_dim
        self.paragraph_out_dim = paragraph_out_dim

        self.user_query_title_fc = nn.Linear(self.paragraph_dim, self.paragraph_out_dim)
        self.user_query_fc = nn.Linear(self.paragraph_dim, self.paragraph_out_dim)
        self.fc = nn.Linear(64, 128)
        self.init_weights()

    def init_weights(self):
        initrange = 1.0/256

        self.user_query_title_fc.weight.data.uniform_(-initrange, initrange)
        self.user_query_title_fc.bias.data.zero_()

        self.user_query_fc.weight.data.uniform_(-initrange, initrange)
        self.user_query_fc.bias.data.zero_()

        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, user_content_info):
        user_query_title = user_content_info[:, 0:256]
        user_query = user_content_info[:, 256:512]

        user_query_title_vec = self.user_query_title_fc(user_query_title)
        user_query_vec = self.user_query_fc(user_query)

        user_combined_features = torch.cat((user_query_vec, user_query_title_vec), 1)
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

        self.fc = nn.Linear(80, 128)
        self.init_weights()

    def init_weights(self):
        initrange = 1.0/256
        self.query_hotrate_embedding.weight.data.uniform_(-initrange, initrange)

        self.query_fc.weight.data.uniform_(-initrange, initrange)
        self.query_fc.bias.data.zero_()

        self.query_title_fc.weight.data.uniform_(-initrange, initrange)
        self.query_title_fc.bias.data.zero_()

        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, query_content_info):
        query_hotrate = query_content_info[:, 0]
        query_hotrate = query_hotrate.reshape(query_hotrate.shape[0], 1)
        query = query_content_info[:, 1:257]
        query_title = query_content_info[:, 257:]
        query_hotrate_embedded = self.query_hotrate_embedding(torch.LongTensor([0])).repeat(
            query_hotrate.shape[0], 1) * (query_hotrate)
        query_hotrate_vec = self.query_hotrate_fc(query_hotrate_embedded)
        query_vec = self.query_fc(query)
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
        user_features = self.user_combined_features_model(X_data[:, :512])
        query_features = self.query_combined_features_model(X_data[:, 512:1025])
        inner_product = torch.sum(user_features.mul(query_features), dim=1)
        print("inner_product=",inner_product)
        result = torch.sigmoid(inner_product)
        return result

def plot_curve(y_test, y_pred):
    auc_score = roc_auc_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    precision, recall, thres = precision_recall_curve(y_test, y_pred)
    # plt.plot(recall,precision)
    F1 = [2 * p * r / (r + p) for (p, r) in zip(precision[:-1], recall[:-1])]
    F1_max_index = F1.index(max(F1))
    # print auc_score
    plt.figure(figsize=(10, 10), dpi=80)
    plt.figure(1)
    ax1 = plt.subplot(221)
    ax1.set_xlabel("fpr")
    ax1.set_ylabel("tpr  and auc_score=%f" % auc_score)
    plt.plot(fpr, tpr, color="r", linestyle="-")
    ax2 = plt.subplot(222)
    ax2.set_xlabel("thres")
    ax2.set_ylabel("precision and recall")
    plt.plot(thres, precision[:-1], '#8B0000', thres, recall[:-1], 'r--')
    ax3 = plt.subplot(223)
    ax3.set_xlabel("thres")
    ax3.set_ylabel("F1 and max_value's thres=%f" % thres[F1_max_index])
    plt.plot(thres, F1, '#9ACD32')
    # plt.plot([thres[F1_max_index],thres[F1_max_index]],[0,F1[F1_max_index]])
    ax4 = plt.subplot(224)
    ax4.set_xlabel("recall")
    ax4.set_ylabel("precision")
    plt.plot(recall, precision, '#9ACD32')
    plt.savefig("roc_F1_prec_recall.jpg")
    plt.show()

def test_loss(LR_model, val_dataloader, criterion):
    targets, predicts = list(), list()
    loss_avg = 0
    with torch.no_grad():
        for step, sample_batched in enumerate(val_dataloader):
            batch = tuple(t for t in sample_batched)

            X_data, Y_data = batch
            print(X_data.shape)
            out = LR_model(X_data)

            out = out.squeeze()
            loss = criterion(out, Y_data.float())
            loss_avg += loss.mean().item()
            targets.extend(Y_data.tolist())
            y = out.squeeze()
            predicts.extend(y.tolist())
        loss_avg = loss_avg / (step + 1)
        try:
            auc = roc_auc_score(targets, predicts)
        except:
            auc = -1
        plot_curve(targets, predicts)
    return loss_avg, auc


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    # filename = "temp_train_data"
    filename = "t"
    dataset = feature_label_DataSet(filename)
    train_num = int(len(dataset) * 0.1)
    test_num = len(dataset) - train_num
    feature_tensor = torch.tensor([f["feature"] for f in dataset], dtype=torch.float)
    label_tensor = torch.tensor([f["label"] for f in dataset], dtype=torch.int)
    dataset_tensor = TensorDataset(feature_tensor, label_tensor)

    train_set, test_set = random_split(dataset_tensor, (train_num, test_num))
    train_dataloader = DataLoader(train_set, batch_size=256, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=2560, shuffle=True)
    user_combined_features_model = user_combined_features()
    query_combined_features_model = query_combined_features([1, 16])
    user_combined_features_model.load_state_dict(torch.load("user_model_state_dict_0.924898",map_location=torch.device('cpu')))
    query_combined_features_model.load_state_dict(torch.load("query_model_state_dict_0.924898",map_location=torch.device('cpu')))
    LR_model = LR_net(user_combined_features_model, query_combined_features_model)
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.BCELoss()
    # LR_model = nn.DataParallel(LR_model, device_ids=[0, 1, 2, 3]).cuda()  # multi-GPU
    loss, auc = test_loss(LR_model, test_dataloader, criterion)
    print("testloss=%f\tauc=%f" % (loss, auc))
