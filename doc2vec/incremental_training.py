from gensim.models.doc2vec import Doc2Vec, TaggedDocument, TaggedLineDocument
import jieba
import time

class DocsLeeCorpus(object):
    def __init__(self, string_tags=False, unicode_tags=False):
        self.string_tags = string_tags
        self.unicode_tags = unicode_tags

    def __iter__(self):
        with open("result_data_cut") as f:
            for i, line in enumerate(f):
                line_list = line.strip().split(" ")
                yield TaggedDocument(line_list, [i])

begin = time.time()
print("begin")
model_file = "model.bin"
train_file = "result_data_cut"
#train_file = "t"
model = Doc2Vec(DocsLeeCorpus(), hs=0, negative=10, dbow_words=1, sample=1e-5,  min_count=2, epochs=200, workers=6)
end = time.time()
print("cost time is %d"%(end-begin))

