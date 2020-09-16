import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, TensorDataset, random_split
import sys

class feature_label_DataSet(IterableDataset):
    def __init__(self, filename, vocab_file, block_size):
        self.filename = filename
        self.block_size = block_size
        self.w2i_dict = self.word2index(vocab_file)
         
    def word2index(self, filename):
        w2i_dict = {}
        with open(filename) as f:
            for line in f:
                line_list = line.strip().split("\t")
                if len(line_list) != 2:
                    continue
                word, index = line_list
                w2i_dict[word] = int(index)
        return w2i_dict 

    def __iter__(self):
        with open(self.filename) as f:
            for index, line in enumerate(f):
                line_list = line.strip().split("\t")
                if len(line_list) != 2:
                    continue
                label, feature = line_list
                label = int(label)
       
                feature_list = [int(ele) for ele in feature.split(" ")]
                label = torch.tensor(label, dtype=torch.long)
                feature_list = torch.tensor(feature_list, dtype=torch.long)
                yield feature_list, label
