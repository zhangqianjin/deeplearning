#coding:utf-8
import argparse
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict

import numpy as np

import faiss
import tensorflow as tf
from data_iterator import DataIterator
from dnn import *
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('-p', type=str, default='test', help='test')
parser.add_argument('--random_seed', type=int, default=19)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--num_interest', type=int, default=4)
parser.add_argument('--learning_rate', type=float, default=0.001, help='')
parser.add_argument('--max_iter', type=int, default=1000, help='(k)')
parser.add_argument('--patience', type=int, default=200)
parser.add_argument('--coef', default=None)
parser.add_argument('--topN', type=int, default=500)
parser.add_argument('--data_path', type=str, default="")
parser.add_argument('--model_path', type=str, default="./model/best_model")
parser.add_argument('--gpu_num', type=str, default="0")
best_metric = 0

def prepare_data(src, target, flag = "train"):
    nick_id, item_id = src
    hist_item, hist_mask = target
    if flag == 'eval':
        item_id = tf.keras.preprocessing.sequence.pad_sequences(item_id, padding="post")
 
    nick_id = tf.convert_to_tensor(nick_id)
    item_id = tf.convert_to_tensor(item_id)
    hist_item = tf.convert_to_tensor(hist_item)
    hist_mask = tf.convert_to_tensor(hist_mask)
    return nick_id, item_id, hist_item, hist_mask

def load_item_cate(source):
    item_cate = {}
    with open(source, 'r') as f:
        for line in f:
            conts = line.strip().split('\t')
            item_id = int(conts[0])
            cate_id = int(conts[1])
            item_cate[item_id] = cate_id
    return item_cate

def load_item_index(source):
    index_item = {}
    with open(source, 'r') as f:
        for line in f:
            conts = line.strip().split('\t')
            item_id = int(conts[0])
            index = int(conts[1])
            index_item[index] = item_id
    return index_item

def load_new_docid(source):
    docid_map = {}
    with open(source, 'r') as f:
        for index, line in enumerate(f):
            if index == 0:
                continue
            line = line.strip()
            docid_map[int(line)] = 0
    return docid_map


def compute_diversity(item_list, item_cate_map):
    n = len(item_list)
    diversity = 0.0
    for i in range(n):
        for j in range(i+1, n):
            diversity += item_cate_map[item_list[i]] != item_cate_map[item_list[j]]
    diversity /= ((n-1) * n / 2)
    return diversity

def evaluate_full(test_data, model, model_path, batch_size, item_cate_map, index_item_map, new_docid_map, save=True, coef=None):
    save_file = "%s/result_data/predict_uid_doclist_%s"%(args.data_path, args.gpu_num)
    print("save_file=", save_file)
    save_f = open(save_file, 'w')
    topN = args.topN
    
    item_embs = model.output_item()

    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0
    try:
        gpu_index = faiss.GpuIndexFlatIP(res, args.embedding_dim, flat_config)
        gpu_index.add(item_embs.numpy())
    except Exception as e:
        return {}
    total = 0
    total_recall = 0.0
    total_ndcg = 0.0
    total_hitrate = 0
    total_map = 0.0
    total_diversity = 0.0
    for src, tgt in test_data:
        nick_id, item_id, hist_item, hist_mask = prepare_data(src, tgt, 'eval')
        if hist_mask.shape[0] < 1:
            continue
        user_embs = model.output_user(hist_item, hist_mask)
        nick_id = nick_id.numpy().tolist()
        if len(user_embs.shape) == 2:
            D, I = gpu_index.search(user_embs.numpy(), int(topN*1.5))
            for i, uid in enumerate(nick_id):
                item_list = I[i]
                cand_item_list = [index_item_map[index] for index in item_list]
                result_item_list = [item for item in cand_item_list if item in new_docid_map]
                result_item_list_str = list(map(str, result_item_list))
                result_str = "%d\t%s"%(uid, ",".join(result_item_list_str[:topN]))
                save_f.write("%s\n"%result_str)
         
        else:
            ni = user_embs.shape[1]
            print("user_embs.shape=", user_embs.shape)
            user_embs = np.reshape(user_embs, [-1, user_embs.shape[-1]])
            print("user_embs.shape=", user_embs.shape)
            print("item_embs.shape=", item_embs.shape)
            D, I = gpu_index.search(user_embs, topN)
            print("I.shape=",I.shape)
            for i, uid in enumerate(nick_id):
                result_item_list = []
                if coef is None:
                    item_list = list(zip(np.reshape(I[i*ni:(i+1)*ni], -1), np.reshape(D[i*ni:(i+1)*ni], -1)))
                    item_list.sort(key=lambda x:x[1], reverse=True)
                    for j in range(len(item_list)):
                        if item_list[j][0] not in result_item_list and item_list[j][0] != 0:
                            if index_item_map[item_list[j][0]] in  new_docid_map: 
                                result_item_list.append(item_list[j][0])
                            if len(result_item_list) >= topN:
                                break
                else:
                    coef = float(coef)
                    origin_item_list = list(zip(np.reshape(I[i*ni:(i+1)*ni], -1), np.reshape(D[i*ni:(i+1)*ni], -1)))
                    origin_item_list.sort(key=lambda x:x[1], reverse=True)
                    item_list = []
                    tmp_item_set = set()
                    for (x, y) in origin_item_list:
                        if x not in tmp_item_set and x in item_cate_map:
                            y = float(y)
                            x = int(x)
                            if x == 0:
                                continue
                            item_list.append((x, y, item_cate_map[x]))
                            tmp_item_set.add(x)
                    cate_dict = defaultdict(int)
                    for j in range(topN):
                        max_index = 0
                        
                        max_score = item_list[0][1] - coef * cate_dict[item_list[0][2]]
                        for k in range(1, len(item_list)):
                            if item_list[k][1] - coef * cate_dict[item_list[k][2]] > max_score:
                                max_index = k
                                max_score = item_list[k][1] - coef * cate_dict[item_list[k][2]]

                        if index_item_map[item_list[max_index][0]] in  new_docid_map:
                            result_item_list.append(item_list[max_index][0])
                            cate_dict[item_list[max_index][2]] += 1
                        item_list.pop(max_index)
                            
                result_list = list(map(str,[index_item_map[index] for index in result_item_list]))
                result_str = "%d\t%s"%(uid, ",".join(result_list))
                save_f.write("%s\n"%result_str)
    save_f.close()


def test(test_file,
        cate_file,
        item_map_file,
        last_new_docid_file,
        item_count,
        batch_size = 1024,
        maxlen = 100,
        lr = 0.001
):
    best_model_path = "%s/model.ckpt"%args.model_path
    # gpu_options = tf.GPUOptions(allow_growth=True)
    item_cate_map = load_item_cate(cate_file)
    index_item_map = load_item_index(item_map_file)
    new_docid_map = load_new_docid(last_new_docid_file)
    model = dnn_model(item_count, args.embedding_dim, args.hidden_size, batch_size, args.num_interest, maxlen)
    model.load_weights(best_model_path)

    test_data = DataIterator(test_file, batch_size, maxlen, train_flag=2)
    evaluate_full(test_data, model, best_model_path, batch_size, item_cate_map, index_item_map, new_docid_map, save=False, coef=args.coef)

def output(
        item_count,
        batch_size = 128,
        maxlen = 100,
        lr = 0.001
):
    best_model_path = "%s/model.ckpt"%args.model_path
    model = dnn_model(item_count, args.embedding_dim, args.hidden_size, batch_size, args.num_interest, maxlen)
    model.load_weights(best_model_path)
    item_embs = model.output_item()
    np.save('output/' + '_emb.npy', item_embs)

if __name__ == '__main__':
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("gpus=", gpus) 
    print(sys.argv)
    SEED = args.random_seed

    # tf.set_random_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    item_count = 2000
    #batch_size = 4096
    batch_size = 4096
    maxlen = 20

    test_file = "%s/predict_data/last_readlist_data_%s"%(args.data_path, args.gpu_num)
    print("test_file=", test_file)
    cate_file = "%s/item_cate.txt"%args.data_path
    item_map_file = "%s/item_map.txt"%args.data_path
    last_new_docid_file = "%s/docid_info"%args.data_path
    test(test_file=test_file, cate_file=cate_file,item_map_file = item_map_file, last_new_docid_file = last_new_docid_file, item_count=item_count, batch_size=batch_size,
             maxlen=maxlen, lr=args.learning_rate)
