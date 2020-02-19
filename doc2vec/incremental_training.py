from gensim.utils import  simple_preprocess
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import jieba

def read_corpus(fname, tokens_only=False):
    with open(fname) as f:
        for ele in f:
            id, title = ele.strip().split("\t")
            tokens = jieba.lcut(title)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield TaggedDocument(tokens, [int(id)])

model_file = "model.bin"
train_file = "train_file"
train_corpus = read_corpus(train_file)
model = Doc2Vec(vector_size=50, min_count=2, epochs=40)
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
model.save(model_file)
