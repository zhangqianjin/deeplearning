from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf 
from tensorflow.keras.layers import Dense, Flatten, Conv2D

from tensorflow.keras import Model
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 添加一个通道维度
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(128).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    #self.d2 = Dense(10, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    #return self.d2(x)
    return x

#gpus = tf.config.experimental.list_physical_devices('GPU')
mirrored_strategy = tf.distribute.MirroredStrategy()

W = tf.Variable(tf.random.normal([10, 128],stddev=1, seed=1))
#b = tf.Variable(tf.zeros([10]))
b = tf.constant(tf.zeros([10]))
with mirrored_strategy.scope(): 
    model = MyModel()
    
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
  with tf.GradientTape(persistent=True) as tape:
    user_emb = model(images)
    
    loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(W, b, tf.reshape(labels,[-1,1]), user_emb, num_sampled=2,num_classes=10))
  gradients = tape.gradient(loss, [W, b])
  optimizer.apply_gradients(zip(gradients,[W,b]))
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients,model.trainable_variables))
  #optimizer.apply_gradients(zip(gradients, [model.trainable_variables, W, b]))

  train_loss(loss)
  logits = tf.matmul(user_emb, tf.transpose(W))
  logits = tf.nn.bias_add(logits, b)
  predictions = tf.nn.softmax(logits)
  train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
  user_emb = model(images, training=False)
  logits = tf.matmul(user_emb, tf.transpose(W))
  logits = tf.nn.bias_add(logits, b)
  predictions = tf.nn.softmax(logits)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

EPOCHS = 50

for epoch in range(EPOCHS):
  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print (template.format(epoch+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         test_loss.result(),
                         test_accuracy.result()*100))
