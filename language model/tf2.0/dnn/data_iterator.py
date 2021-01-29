import numpy
import json
import random
import numpy as np


class DataIterator:

    def __init__(self, source,
                 batch_size=128,
                 maxlen=100,
                 train_flag=0
                ):
        self.read(source)
        self.users = list(self.users)
        
        self.batch_size = batch_size
        self.eval_batch_size = batch_size
        self.train_flag = train_flag
        self.maxlen = maxlen
        self.index = 0
        self.total_user = len(self.users)

    def __iter__(self):
        return self
    
    def next(self):
        return self.__next__()

    def read(self, source):
        self.graph = {}
        self.users = []
        num = 0
        with open(source, 'r') as f:
            for line in f:
                num += 1
                #print("%s %d" % (source, num))
                conts = line.strip().split('\t')
                if len(conts) != 2:
                    continue
                uid, doc_str = conts
                   
                user_id = int(uid)
                doc_str_list = doc_str.split(",")
                if len(doc_str_list) < 10:
                    continue
                self.graph[user_id] = list(map(int, doc_str_list))
                self.users.append(user_id)
    
    def __next__(self):
        if self.train_flag == 0:
            #user_id_list = random.sample(self.users, self.batch_size)
            if self.index >= self.total_user:
                self.index = 0
                raise StopIteration
            user_id_list = self.users[self.index: self.index+self.batch_size]
            self.index += self.batch_size
        else:
            #total_user = len(self.users)
            if self.index >= self.total_user:
                self.index = 0
                raise StopIteration
            user_id_list = self.users[self.index: self.index+self.eval_batch_size]
            self.index += self.eval_batch_size

        item_id_list = []
        hist_item_list = []
        hist_mask_list = []
        for user_id in user_id_list:
            item_list = self.graph[user_id]
            item_list_len = len(item_list)
            if self.train_flag == 0:
                k = random.choice(range(4, item_list_len))
                item_id_list.append(item_list[k])
            else:
                k = int(item_list_len * 0.8)
                item_id_list.append(item_list[k:])

            if k >= self.maxlen:
                hist_item_list.append(item_list[k-self.maxlen: k])
                hist_mask_list.append([1.0] * self.maxlen)
            else:
                hist_item_list.append(item_list[:k] + [0] * (self.maxlen - k))
                hist_mask_list.append([1.0] * k + [0.0] * (self.maxlen - k))        
        return (user_id_list, item_id_list), (hist_item_list, hist_mask_list)

