# coding=utf-8
from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import sys
import random
from tqdm import tqdm, trange
import torchvision.models as cv_models
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,random_split,TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss, MSELoss

from tensorboardX import SummaryWriter

from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification,BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from train_processor import text_corrProcessor, convert_examples_to_features
from sklearn.metrics import roc_auc_score
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


logger = logging.getLogger(__name__)

def fetch_bert_model(model_path):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path)
    for para in model.parameters():
        para.requires_grad=False
    model.cuda()
    return model, tokenizer 

def fetch_imgid_feature(filename):
    imgid_vec_dict = {}
    with open(filename) as f:
        for line in f:
            line_list = line.strip().split("\t")
            if len(line_list)<0:
                continue
            index, feature_str = line_list
            index = int(index)
            feature_list = []
            for ele in feature_str.split(","):
                feature_list.append(float(ele))
            imgid_vec_dict[index] = feature_list
    return imgid_vec_dict

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc = nn.Linear(1280, 1)
        self.sg = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sg(x)
        return x


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    parser.add_argument("--val_data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")



    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--train_img_path",
                        default=None,
                        type=str,
                        required=True,
                        help="train img feature path.")

    parser.add_argument("--val_img_path",
                        default=None,
                        type=str,
                        required=True,
                        help="train img feature path.")



    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir',
                        action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    args.device = device

    processor = text_corrProcessor()

    bert_model, tokenizer = fetch_bert_model(args.bert_model) 

        # Prepare data loader
    train_imgid_vec_dict = fetch_imgid_feature(args.train_img_path)
    val_imgid_vec_dict = fetch_imgid_feature(args.val_img_path)
     
    train_examples = processor.get_train_examples(args.train_data_dir)
    train_features = convert_examples_to_features(
    train_examples, train_imgid_vec_dict, args.max_seq_length, tokenizer)
    train_all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    train_all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    train_all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    train_all_imgvec = torch.tensor([f.imgvec for f in train_features], dtype=torch.float)
    train_all_label = torch.tensor([f.label for f in train_features], dtype=torch.float)
    
    train_data = TensorDataset(train_all_input_ids, train_all_input_mask, train_all_segment_ids,train_all_imgvec, train_all_label)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.train_batch_size)

    val_examples = processor.get_train_examples(args.val_data_dir)
    val_features = convert_examples_to_features(
    val_examples, val_imgid_vec_dict, args.max_seq_length, tokenizer)
    val_all_input_ids = torch.tensor([f.input_ids for f in val_features], dtype=torch.long)
    val_all_input_mask = torch.tensor([f.input_mask for f in val_features], dtype=torch.long)
    val_all_segment_ids = torch.tensor([f.segment_ids for f in val_features], dtype=torch.long)
    val_all_imgvec = torch.tensor([f.imgvec for f in val_features],dtype=torch.float)
    val_all_label = torch.tensor([f.label for f in val_features], dtype=torch.float)
    val_data = TensorDataset(val_all_input_ids, val_all_input_mask, val_all_segment_ids, val_all_imgvec,val_all_label)
    val_dataloader = DataLoader(val_data, shuffle=False, batch_size=args.train_batch_size)
    LR_model = Net()
    LR_model.load_state_dict(torch.load("model/0_1800_0.501381_model_param")) 
    LR_model = LR_model.cuda()
    criterion = nn.BCELoss()
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(LR_model.parameters(), lr=1e-3)
    best_auc = 0.501381
    for epoch in range(10000):
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids,imgvec, label = batch

            all_encoder_layers, pool_output = bert_model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
            merge_feature = torch.cat((pool_output, imgvec),1)
            out = LR_model(merge_feature)
            out = out.squeeze()
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("epoch=%d\ti_batch=%d\tloss=%f"%(epoch,step,loss.item()))
            if (1+step)%1000 == 0:
                torch.save(LR_model.state_dict(),"model/%d_%d_%f_model_param"%(epoch,step,loss.item()))

def test(bert_model,LR_model,val_dataloader):
    LR_model.eval()
    targets, predicts = list(), list()
    for step, batch in enumerate(tqdm(val_dataloader, desc="Iteration")):
        batch = tuple(t.cuda() for t in batch)
        input_ids, input_mask, segment_ids,imgvec, label = batch
        all_encoder_layers, pool_output = bert_model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        merge_feature = torch.cat((pool_output, imgvec),1)
        y = LR_model(merge_feature)
        targets.extend(label.tolist())
        predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts)
         


if __name__ == "__main__":
    main()
