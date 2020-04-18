import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, TensorDataset, random_split
import sys

class feature_label_DataSet(IterableDataset):
    def __init__(self, filename, block_size):
        self.filename = filename
        self.block_size = block_size

    def __iter__(self):
        block_list = []
        with open(self.filename) as f:
            for index, line in enumerate(f):
                line_list = line.strip().split(",")
                if len(line_list) < 259:
                    continue
                feature_str = line_list[:-1]
                label_str = line_list[-1]
                feature = list(map(float,feature_str))
                label = float(label_str)
                block_list.append([feature, label])
                if (index+1) % self.block_size == 0:
                    yield block_list
                    block_list = []
        if len(block_list) > 0:
            yield block_list


if __name__ == '__main__':
    filename = sys.argv[1]
    dataset = feature_label_DataSet(filename, block_size=10)
    for block_data in dataset:
        feature_tensor = torch.tensor([f[0] for f in block_data], dtype=torch.float)
        label_tensor = torch.tensor([f[1] for f in block_data], dtype=torch.int)
        dataset_tensor = TensorDataset(feature_tensor, label_tensor)
        train_dataloader = DataLoader(dataset_tensor, batch_size=4, shuffle=True)
        for i_batch, sample_batched in enumerate(train_dataloader):
            batch = tuple(t for t in sample_batched)
            # batch = tuple(t.cuda() for t in sample_batched)
            feature, label = batch
            print(feature)
            print(label.shape)

