import sys
import jieba
from gensim.models.doc2vec import Doc2Vec
import numpy as np
import multiprocessing as mul

model_file = sys.argv[1]
model = Doc2Vec.load(model_file)

def readfile(filename):
    count = 0
    block_list = []
    block_size = 200000
    with open(filename) as f:
        for line in f:
            count += 1
            line = line.strip()
            content_split_list = list(jieba.cut(line))
            if count < block_size:
                block_list.append([line, content_split_list])
            else:
                count = 0
                yield block_list
                block_list = []
        if len(block_list) > 0:
            yield block_list

def multipro_block(data):
    global model
    title, content_split_list = data
    temp_list = []
    title_vec = model.infer_vector(content_split_list)
    result_str = list(map(str, title_vec))
    return "%s\t%s"%(title, ",".join(result_str))

def process():
    filename = sys.argv[2]
    data_stream = readfile(filename)
    cores = mul.cpu_count()
    cores_num = int(cores*0.5)
    with mul.Pool(cores_num) as pool:
        for block in data_stream:
            iter_test = pool.imap(multipro_block, block)
            for e in iter_test:
                print(e)

if __name__ == '__main__':
    process()            
