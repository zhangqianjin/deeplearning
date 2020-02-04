from annoy import AnnoyIndex
import random

def int_2_str(index_list):
    vec_list = []
    for ele in index_list:
        vec_list.append(str(ele))
    return vec_list


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
            query, vec_str = line_list
            vec = str_2_float(vec_str)
            index_vec_list.append([query, vec])
            count += 1
            if count % batch_size ==0:
                yield index_vec_list
                index_vec_list = []
                 


def build_annoy(index_vec_data, feature_len, distance, index_filename):
    annoy_instance = AnnoyIndex(feature_len, distance)  # Length of item vector that will be indexed
    annoy_instance.load(index_filename)
    for data in index_vec_data:
        for query, vec in data:
           nn_index_list =  annoy_instance.get_nns_by_vector(vec,100)
           index_str=",".join(int_2_str(nn_index_list))
           print("%s\t%s"%(query,index_str))


def process():
    filename = "t"
    index_filename = "annoy_instance.ann"
    feature_len = 128
    distance = "dot"
    index_vec_data = readfile(filename)
    index_hotword_dict = fetch_index_query(hotword_file) 
    build_annoy(index_vec_data, feature_len, distance, index_filename)

if __name__ == '__main__':
    process()
