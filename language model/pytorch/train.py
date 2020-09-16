import time
import torch
from torch.optim import SGD, Adam, Adadelta
import torch.nn as nn
from processing import feature_label_DataSet
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
import sys
import math
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch.nn.functional as F
#from encoding.parallel import DataParallelModel,DataParallelCriterion
from parallel import DataParallelModel, DataParallelCriterion

class DM(nn.Module):
    def __init__(self, vocab_size=10000, embedding_size=64, output_size = 2000000):
        super(DM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.fc = nn.Linear(self.embedding_size, self.output_size)
        self.init_weights()
        self.criterion = nn.CrossEntropyLoss()  

    def init_weights(self):
        initrange = 1.0/math.sqrt(self.embedding_size)
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
      

    def forward(self, x, y):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        loss = self.criterion(x,y)
        return loss


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    vocab_file = sys.argv[1]
    data_file = sys.argv[2]
    vocab_size = 100000
    vec_size = 128
    output_size = 1000000
    num_epochs = 20
    batch_size = 1024
    print("begin set embedding")
    model = DM(vocab_size, vec_size, output_size).cuda()
    #model=nn.DataParallel(model, device_ids=[0,1,2,3,4,5,6,7]).cuda()
    #model =DataParallelModel(model)
    #optimizer = Adam(params=model.parameters(), lr=0.01)
    #optimizer = SGD(params=model.parameters(), lr=0.001, momentum=0.95)
    optimizer = Adadelta(params=model.parameters(), lr=1)
    model = nn.DataParallel(model)
    print("model init end and begin train")
    best_auc = 0
    test_loss_list = []
    num = 0
    flag = True
    epoch = 0
    accumulation_steps = 128
    while epoch<100:
        epoch += 1
        print("feature_label_DataSet begin")
        dataset = feature_label_DataSet(data_file,  vocab_file, block_size=1024*16)
        print("feature label tensor begin")
        #feature_tensor = torch.tensor([f[0] for f in block_data], dtype=torch.long)
        #label_tensor = torch.tensor([f[1] for f in block_data], dtype=torch.long)
        
        #dataset_tensor = TensorDataset(feature_tensor, label_tensor)
        train_dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=20)
        for i_batch, sample_batched in enumerate(train_dataloader):
            num += 1
            #batch = tuple(t for t in sample_batched)
            batch = tuple(t.cuda() for t in sample_batched)
            X_data, Y_data = batch
                
            loss = model(X_data, Y_data)
            loss_mean = loss.mean()/accumulation_steps
            loss_mean.backward()
            if num % accumulation_steps ==0:
                #optimizer.zero_grad()
                optimizer.step()
                optimizer.zero_grad()
                print("epoch=%d\tbatch=%d\tloss=%f" % (epoch, i_batch, loss_mean.item()))
        torch.save(model.module.state_dict(), "model/DM_state_dict_%d"%epoch)
