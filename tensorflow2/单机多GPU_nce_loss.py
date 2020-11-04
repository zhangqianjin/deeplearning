from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf 
from tensorflow.keras.layers import Dense, Flatten, Conv2D

from tensorflow.keras import Model

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    #self.d2 = Dense(10, activation='softmax')
    #self.d2 = Dense(10, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    #return self.d2(x)
    return x


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        optimizer = tf.keras.optimizers.SGD()
        model = MyModel()
        W = tf.Variable(tf.random.normal([10, 128],stddev=1, seed=1))
        b = tf.Variable(tf.zeros([10]))
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    GLOBAL_BATCH_SIZE = 128 
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(GLOBAL_BATCH_SIZE)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(GLOBAL_BATCH_SIZE)
    train_dist_dataset = mirrored_strategy.experimental_distribute_dataset(train_ds)
    test_dist_dataset = mirrored_strategy.experimental_distribute_dataset(test_ds)
    with mirrored_strategy.scope():
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

        @tf.function
        def train_step(inputs):
            images, labels = inputs

            with tf.GradientTape() as tape:
                user_emb = model(images, training=True)
                #loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(W, b, tf.reshape(labels,[-1,1]), user_emb, num_sampled=2,num_classes=10))
                loss = tf.reduce_mean(tf.nn.nce_loss(W, b, tf.reshape(labels,[-1,1]), user_emb, num_sampled=2,num_classes=10, remove_accidental_hits=True))

            logits = tf.matmul(user_emb, tf.transpose(W))
            logits = tf.nn.bias_add(logits, b)
            predictions = tf.nn.softmax(logits)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_accuracy.update_state(labels, predictions)
            return loss 
     
        @tf.function
        def test_step(inputs):
            images, labels = inputs

            user_emb = model(images, training=False)
            logits = tf.matmul(user_emb, tf.transpose(W))
            logits = tf.nn.bias_add(logits, b)
            predictions = tf.nn.softmax(logits)
            t_loss = loss_object(labels, predictions)

            test_loss.update_state(t_loss)
            test_accuracy.update_state(labels, predictions)

    with mirrored_strategy.scope():
        @tf.function
        def distributed_train_step(dataset_inputs):
            per_replica_losses = mirrored_strategy.experimental_run_v2(train_step,args=(dataset_inputs,))
            return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,axis=None)
 
        @tf.function
        def distributed_test_step(dataset_inputs):
            mirrored_strategy.experimental_run_v2(test_step, args=(dataset_inputs,))

        for epoch in range(20):
            total_loss = 0.0
            num_batches = 0
            for x in train_dist_dataset:
                total_loss += distributed_train_step(x)
                num_batches += 1
            train_loss = total_loss / num_batches

            for x in test_dist_dataset:
                distributed_test_step(x)
            template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, " "Test Accuracy: {}")
            print(template.format(epoch+1, train_loss,train_accuracy.result()*100, test_loss.result(),test_accuracy.result()*100))

            test_loss.reset_states()
            train_accuracy.reset_states()
            test_accuracy.reset_states()
