import torch
from torch.utils.data import DataLoader, Dataset,TensorDataset,random_split
import sys


class img_text_label_featureDataSet(Dataset):
    def __init__(self,filename):
       self.filename = filename
       content_list = []
       with open(filename) as f:
           for line in f:
               line = line.strip()
               content_list.append(line)
       self.content_info = content_list
     
    def __len__(self):
         return len(self.content_info)

    def __getitem__(self, idx):
        content = self.content_info[idx]
        label, feature = content.split("\t")
        label = int(label)
        feature_list = feature.split(",")
        result_feature = []
        for ele in feature_list:
            result_feature.append(float(ele))
        return {"feature":result_feature,"label":label}


if __name__ == '__main__':
    filename = sys.argv[1]
    dataset = img_text_label_featureDataSet(filename)
    train_num = int(len(dataset)*0.8)
    test_num = len(dataset) - train_num
    print("test_num=%d"%test_num)
    feature_tensor =  torch.tensor([f["feature"] for f in dataset], dtype=torch.float)
    label_tensor = torch.tensor([f["label"] for f in dataset], dtype=torch.int)
    dataset_tensor = TensorDataset(feature_tensor, label_tensor)
    
    train_set, test_set = random_split(dataset_tensor, (train_num, test_num))
    train_dataloader = DataLoader(train_set, batch_size=4,shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=4,shuffle=True)
    for i_batch, sample_batched in enumerate(test_dataloader):
        print("i_bath=%d\tsample_batched=%d"%(i_batch,sample_batched[0].shape[0]))
        batch = tuple(t.cuda() for t in sample_batched)
        feature, label = batch
        print(feature)
        print(label)
        print("....................")        








