from annoy import AnnoyIndex
import random

def str_2_float(vec_str):
    vec_list = []
    for ele in vec_str.split(","):
        vec_list.append(float(ele))
    return vec_list

def readfile(filename, batch_size=1000):
    index_vec_list = []
    count = 0
    with open(filename) as f:
        for line in f:
            line_list = line.strip().split("\t")
            index_str,vec_str = line_list
            index = int(index_str)
            vec = str_2_float(vec_str)
            index_vec_list.append([index, vec])
            count += 1
            if count % batch_size == 0:
                yield index_vec_list
                index_vec_list = [] 


def build_annoy(index_vec_data, feature_len, distance):
    annoy_instance = AnnoyIndex(feature_len, distance)  # Length of item vector that will be indexed
    for data in index_vec_data:
        for index, vec in data:
            annoy_instance.add_item(index, vec)

    annoy_instance.build(20) # 10 trees
    annoy_instance.save('annoy_instance.ann')

def process():
    filename = "t"
    feature_len = 128
    distance = "dot"
    index_vec_data = readfile(filename)
    build_annoy(index_vec_data, feature_len, distance)

if __name__ == '__main__':
    process()
