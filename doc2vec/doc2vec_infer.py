from gensim.utils import  simple_preprocess
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import jieba
import gensim

def read_corpus_dict(fname, tokens_only=False):
    docid_dict = {}
    with open(fname) as f:
        for ele in f:
            id, title = ele.strip().split("\t")
            tokens = jieba.lcut(title)
            docid_dict[int(id)] = tokens
    return docid_dict

model_file = "model.bin"
train_file = "train_file"
model = Doc2Vec.load(model_file,mmap='r')
docid_word_dict = read_corpus_dict(train_file)

ranks = []
second_ranks = []
for doc_id in docid_word_dict:
    print(docid_word_dict[doc_id])
    inferred_vector = model.infer_vector(docid_word_dict[doc_id])
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)

    second_ranks.append(sims[1])

print(ranks)
