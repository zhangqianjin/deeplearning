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
from ComiRec_SA import *
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('-p', type=str, default='train', help='train | test')
parser.add_argument('--dataset', type=str, default='book', help='book | taobao')
parser.add_argument('--random_seed', type=int, default=19)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--num_interest', type=int, default=4)
parser.add_argument('--model_type', type=str, default='ComiRec-SA', help='DNN | GRU4REC | ..')
parser.add_argument('--learning_rate', type=float, default=0.001, help='')
parser.add_argument('--max_iter', type=int, default=1000, help='(k)')
parser.add_argument('--patience', type=int, default=50)
parser.add_argument('--coef', default=None)
parser.add_argument('--topN', type=int, default=50)

best_metric = 0

def prepare_data(src, target):
    nick_id, item_id = src
    hist_item, hist_mask = target
    
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

def compute_diversity(item_list, item_cate_map):
    n = len(item_list)
    diversity = 0.0
    for i in range(n):
        for j in range(i+1, n):
            diversity += item_cate_map[item_list[i]] != item_cate_map[item_list[j]]
    diversity /= ((n-1) * n / 2)
    return diversity

def evaluate_full(test_data, model, model_path, batch_size, item_cate_map, save=True, coef=None):
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
        nick_id, item_id, hist_item, hist_mask = prepare_data(src, tgt)
        if hist_mask.shape[0] < 1:
            continue
        user_embs = model.output_user(hist_item, hist_mask)
        item_id = item_id.numpy().tolist()
        if len(user_embs.shape) == 2:
            D, I = gpu_index.search(user_embs.numpy(), topN)
            for i, iid_list in enumerate(item_id):
                recall = 0
                dcg = 0.0
                item_list = set(I[i])
                for no, iid in enumerate(iid_list):
                    if iid in item_list:
                        recall += 1
                        dcg += 1.0 / math.log(no+2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no+2, 2)
                total_recall += recall * 1.0 / len(iid_list)
                if recall > 0:
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
                if not save:
                    total_diversity += compute_diversity(I[i], item_cate_map)
        else:
            ni = user_embs.shape[1]
            user_embs = np.reshape(user_embs, [-1, user_embs.shape[-1]])
            D, I = gpu_index.search(user_embs, topN)
            for i, iid_list in enumerate(item_id):
                recall = 0
                dcg = 0.0
                item_list_set = set()
                if coef is None:
                    item_list = list(zip(np.reshape(I[i*ni:(i+1)*ni], -1), np.reshape(D[i*ni:(i+1)*ni], -1)))
                    item_list.sort(key=lambda x:x[1], reverse=True)
                    for j in range(len(item_list)):
                        if item_list[j][0] not in item_list_set and item_list[j][0] != 0:
                            item_list_set.add(item_list[j][0])
                            if len(item_list_set) >= topN:
                                break
                else:
                    origin_item_list = list(zip(np.reshape(I[i*ni:(i+1)*ni], -1), np.reshape(D[i*ni:(i+1)*ni], -1)))
                    origin_item_list.sort(key=lambda x:x[1], reverse=True)
                    item_list = []
                    tmp_item_set = set()
                    for (x, y) in origin_item_list:
                        if x not in tmp_item_set and x in item_cate_map:
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
                            elif item_list[k][1] < max_score:
                                break
                        item_list_set.add(item_list[max_index][0])
                        cate_dict[item_list[max_index][2]] += 1
                        item_list.pop(max_index)

                for no, iid in enumerate(iid_list):
                    if iid in item_list_set:
                        recall += 1
                        dcg += 1.0 / math.log(no+2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no+2, 2)
                total_recall += recall * 1.0 / len(iid_list)
                if recall > 0:
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
                if not save:
                    total_diversity += compute_diversity(list(item_list_set), item_cate_map)
        
        total += len(item_id)
    
    recall = total_recall / total
    ndcg = total_ndcg / total
    hitrate = total_hitrate * 1.0 / total
    diversity = total_diversity * 1.0 / total
    if save:
        return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate}
    return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate, 'diversity': diversity}



def train(train_file,
        valid_file,
        test_file,
        cate_file,
        item_count,
        dataset = "book",
        batch_size = 128,
        maxlen = 100,
        test_iter = 50,
        model_type = 'DNN',
        lr = 0.001,
        max_iter = 100,
        patience = 20
):

    best_model_path = "best_model/model.ckpt"

    # gpu_options = tf.GPUOptions(allow_growth=True)

    writer = SummaryWriter('runs/')

    item_cate_map = load_item_cate(cate_file)

    train_data = DataIterator(train_file, batch_size, maxlen, train_flag=0)
    valid_data = DataIterator(valid_file, batch_size, maxlen, train_flag=1)
        
    model = ComiRec_SA(item_count, args.embedding_dim, args.hidden_size, batch_size, args.num_interest, maxlen)
    optimizer = tf.keras.optimizers.SGD()
    print('training begin')
    sys.stdout.flush()

    start_time = time.time()
    iter = 0
    try:
        loss_sum = 0.0
        trials = 0

        for src, tgt in train_data:
            nick_id, item_id, hist_item, hist_mask = prepare_data(src, tgt)
    
            with tf.GradientTape(persistent=False) as tape:
                loss = model(item_id, hist_item, hist_mask)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables)) 
            loss_sum += loss
            iter += 1

            if iter % test_iter == 0:
               
                print("begin evaluate_full iter=%d"%iter)
                metrics = evaluate_full(valid_data, model, best_model_path, batch_size, item_cate_map)
                print("end evaluate_full")
                log_str = 'iter: %d, train loss: %.4f' % (iter, loss_sum / test_iter)
                if metrics != {}:
                    log_str += ', ' + ', '.join(['valid ' + key + ': %.6f' % value for key, value in metrics.items()])

                print(log_str)
                writer.add_scalar('train/loss', loss_sum.numpy() / test_iter, iter)
                if metrics != {}:
                    for key, value in metrics.items():
                        writer.add_scalar('eval/' + key, value, iter)
                    
                if 'recall' in metrics:
                    recall = metrics['recall']
                    global best_metric
                    if recall > best_metric:
                        best_metric = recall
                        model.save_weights(best_model_path)
                        trials = 0
                    else:
                        trials += 1
                        if trials > patience:
                            break

                loss_sum = 0.0
                test_time = time.time()
                print("time interval: %.4f min" % ((test_time-start_time)/60.0))
                sys.stdout.flush()

                if iter >= max_iter * 1000:
                    break
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
    print("begin load")
    save_model = ComiRec_SA(item_count, args.embedding_dim, args.hidden_size, batch_size, args.num_interest, maxlen)
    save_model.load_weights(best_model_path)
    print(save_model)
    print("end load")
    metrics = evaluate_full(valid_data, save_model, best_model_path, batch_size, item_cate_map, save=False)
    print(', '.join(['valid ' + key + ': %.6f' % value for key, value in metrics.items()]))

    test_data = DataIterator(test_file, batch_size, maxlen, train_flag=2)
    metrics = evaluate_full(test_data, save_model, best_model_path, batch_size, item_cate_map, save=False)
    print(', '.join(['test ' + key + ': %.6f' % value for key, value in metrics.items()]))

def test(test_file,
        cate_file,
        item_count,
        dataset = "book",
        batch_size = 128,
        maxlen = 100,
        model_type = 'DNN',
        lr = 0.001
):
    best_model_path = "best_model/model.ckpt"
    # gpu_options = tf.GPUOptions(allow_growth=True)
    item_cate_map = load_item_cate(cate_file)
    model = ComiRec_SA(item_count, args.embedding_dim, args.hidden_size, batch_size, args.num_interest, maxlen)
    model.load_weights(best_model_path)
        
    test_data = DataIterator(test_file, batch_size, maxlen, train_flag=2)
    metrics = evaluate_full(test_data, model, best_model_path, batch_size, item_cate_map, save=False, coef=args.coef)
    print(', '.join(['test ' + key + ': %.6f' % value for key, value in metrics.items()]))

def output(
        item_count,
        dataset = "book",
        batch_size = 128,
        maxlen = 100,
        model_type = 'DNN',
        lr = 0.001
):
    best_model_path = "best_model/model.ckpt"
    # gpu_options = tf.GPUOptions(allow_growth=True)
    model = ComiRec_SA(item_count, args.embedding_dim, args.hidden_size, batch_size, args.num_interest, maxlen)
    model.load_weights(best_model_path)
    item_embs = model.output_item()
    np.save('output/' + '_emb.npy', item_embs)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True) 
    print(sys.argv)
    args = parser.parse_args()
    SEED = args.random_seed

    # tf.set_random_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    train_name = 'train'
    valid_name = 'valid'
    test_name = 'test'
    """
    if args.dataset == 'taobao':
        path = './data/taobao_data/'
        item_count = 1708531
        batch_size = 256
        maxlen = 50
        test_iter = 500
    elif args.dataset == 'book':
        path = './data/book_data/'
        item_count = 367983
        batch_size = 128
        maxlen = 20
        test_iter = 1000
    
    train_file = path + args.dataset + '_train.txt'
    valid_file = path + args.dataset + '_valid.txt'
    test_file = path + args.dataset + '_test.txt'
    cate_file = path + args.dataset + '_item_cate.txt'
    dataset = args.dataset
    """
    item_count = 2484843
    batch_size = 4096
    maxlen = 20
    test_iter = 500
    train_file = "data/2020-12-02/train.txt"
    valid_file = "data/2020-12-02/valid.txt"
    test_file = "data/2020-12-02/test.txt"
    cate_file = "data/2020-12-02/item_cate.txt"
    #train_file = "%s/train.txt"%path
    #valid_file = "%s/valid.txt"%path
    #test_file = "%s/test.txt"%path
    #cate_file = "%s/item_cate.txt"%path
    dataset = args.dataset

    if args.p == 'train':

        train(train_file=train_file, valid_file=valid_file, test_file=test_file, cate_file=cate_file,
              item_count=item_count, dataset=dataset, batch_size=batch_size, maxlen=maxlen, test_iter=test_iter, 
              model_type=args.model_type, lr=args.learning_rate, max_iter=args.max_iter, patience=args.patience)
    elif args.p == 'test':
        test(test_file=test_file, cate_file=cate_file, item_count=item_count, dataset=dataset, batch_size=batch_size,
             maxlen=maxlen, model_type=args.model_type, lr=args.learning_rate)
    elif args.p == 'output':
        output(item_count=item_count, dataset=dataset, batch_size=batch_size, maxlen=maxlen, 
               model_type=args.model_type, lr=args.learning_rate)
    else:
        print('do nothing...')
