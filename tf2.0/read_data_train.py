import tensorflow as tf

class myModel(tf.keras.Model):
    def __init__(self, vocab_size, output_size, vec_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.vec_size = vec_size
        self.Embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=vec_size)
        self.Dense = tf.keras.layers.Dense(output_size, activation='softmax')

    def call(self, x):
        print(x)
        x = self.Embedding(x)
        x = tf.math.reduce_mean(x, axis=1)
        x = self.Dense(x)
        return x


vocab_size = 10
vec_size = 128
output_size = 10
num_epochs = 10
batch_size = 16

model = myModel(vocab_size, output_size, vec_size)
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


filename = "t"
#titanic_lines = tf.data.TextLineDataset(filename)
def data_func(line):
    line = tf.strings.split(line, sep = ",")
    return line

#titanic_data = titanic_lines.skip(1).map(data_func).map(lambda x: ).map(lambda x:(x[:-1], x[-1]))
dataset = tf.data.TextLineDataset(filename)
dataset = dataset.map(lambda string: tf.strings.split(string, sep=","))
dataset = dataset.map(lambda string : tf.compat.v1.string_to_number(string,tf.int32))
dataset = dataset.map(lambda x: (x[:-1], x[-1]))
dataset = dataset.batch(batch_size=5)
iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
features, labels = iterator.get_next()
model.fit(features, labels, epochs=5)
