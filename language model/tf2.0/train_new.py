
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
    
def input_fn(data_file, batch_size, num_epoch = None):
    print("num_epoch=",num_epoch)
    dataset = (tf.data.TextLineDataset(data_file))
    dataset = dataset.map(lambda string: tf.strings.split(string, sep=","))
    dataset = dataset.map(lambda string : tf.compat.v1.string_to_number(string,tf.int32))
    dataset = dataset.map(lambda x: (x[:-1], x[-1]))
    dataset = dataset.repeat(num_epoch)
    dataset = dataset.batch(batch_size)
    return dataset 



vocab_size = 10
vec_size = 128
output_size = 10
num_epochs = 10
batch_size = 16
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = myModel(vocab_size, output_size, vec_size)
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    
filename = "t"
dataset = input_fn(filename, batch_size=5, num_epoch = 10)
model.fit(dataset, epochs=2, workers=10, batch_size=1024)
