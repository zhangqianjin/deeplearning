import tensorflow as tf
import os
import time


class myModel(tf.keras.Model):
    def __init__(self, vocab_size, output_size, vec_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.vec_size = vec_size
        self.Embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=vec_size)
        self.Line_1 = tf.keras.layers.Dense(self.vec_size, activation='relu')

    def call(self, x):
        x = self.Embedding(x)
        x = tf.math.reduce_mean(x, axis=1)
        x = self.Line_1(x)
        return x

def parse_line(line):
    features={
            "x":tf.io.FixedLenFeature([10],tf.int64),
            "y":tf.io.FixedLenFeature([1],tf.int64)
            }
    return tf.io.parse_single_example(line, features)

def input_fn(data_file, batch_size, num_workers, num_epoch = None):
    dataset = tf.data.TFRecordDataset(data_file, num_parallel_reads=9).map(parse_line,num_parallel_calls = tf.data.experimental.AUTOTUNE).batch(batch_size).repeat(5).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset 



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    #mirrored_strategy = tf.distribute.MirroredStrategy()
    mirrored_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    vocab_size = 1500
    vec_size = 32
    output_size = 2000
    num_epochs = 10
    sample_class = 10
    GLOBAL_BATCH_SIZE = 1024*8
    num_workers = tf.data.experimental.AUTOTUNE 
    with mirrored_strategy.scope():
        optimizer = tf.keras.optimizers.SGD()
        model = myModel(vocab_size, output_size, vec_size)
        W = tf.Variable(tf.random.normal([output_size, vec_size],stddev=1, seed=1))
        b = tf.constant(tf.zeros([output_size]))

    test_file = "test.tfrecord"
    train_file = ["train_00.tfrecord", "train_01.tfrecord"]
    test_file = ["test_00.tfrecord", "test_01.tfrecord"]
    train_ds = input_fn(train_file, GLOBAL_BATCH_SIZE, num_workers, 10)
    test_ds = input_fn(test_file, 128, num_workers, 0)
    train_dist_dataset = mirrored_strategy.experimental_distribute_dataset(train_ds)
    test_dist_dataset = mirrored_strategy.experimental_distribute_dataset(test_ds)
    with mirrored_strategy.scope():
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

        @tf.function
        def train_step(images, labels):
            #images, labels = inputs
            #images, labels = inputs['x'], inputs['y']
            with tf.GradientTape(persistent=True) as tape:
                user_emb = model(images, training=True)
                loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(W, b, labels, user_emb, num_sampled=sample_class,num_classes=output_size))
                #loss = tf.reduce_mean(tf.nn.nce_loss(W, b, tf.reshape(labels,[-1,1]), user_emb, num_sampled=5,num_classes=10, remove_accidental_hits=True))

            #logits = tf.matmul(user_emb, tf.transpose(W))
            #logits = tf.nn.bias_add(logits, b)
            #predictions = tf.nn.softmax(logits)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            gradients = tape.gradient(loss, [W])
            optimizer.apply_gradients(zip(gradients, [W]))
            #train_accuracy.update_state(labels, predictions)
            return loss 
     
        @tf.function
        def test_step(inputs):
            #images, labels = inputs
            images, labels = inputs['x'], inputs['y']
            user_emb = model(images, training=False)
            logits = tf.matmul(user_emb, tf.transpose(W))
            #logits = tf.nn.bias_add(logits, b)
            predictions = tf.nn.softmax(logits)
            t_loss = loss_object(labels, predictions)

            test_loss.update_state(t_loss)
            test_accuracy.update_state(labels, predictions)

    with mirrored_strategy.scope():
        @tf.function
        def distributed_train_step(x, y):
            #per_replica_losses = mirrored_strategy.experimental_run_v2(train_step,args=(dataset_inputs,))
            per_replica_losses = mirrored_strategy.run(train_step,args=(x,y))
            return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,axis=None)
 
        @tf.function
        def distributed_test_step(dataset_inputs):
            #mirrored_strategy.experimental_run_v2(test_step, args=(dataset_inputs,))
            mirrored_strategy.run(test_step, args=(dataset_inputs,))

        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0
            begin_time = time.time()
            for inputs in train_dist_dataset:
                images, labels = inputs['x'], inputs['y'] 
                total_loss += distributed_train_step(images, labels)
                num_batches += 1
                print("%d\t%d\t%f"%(epoch, num_batches, total_loss/num_batches))

            """
            train_loss = total_loss / num_batches

            for x in test_dist_dataset:
                distributed_test_step(x)
            print("%d\t%f\t%f\t%f"%(epoch+1, train_loss, test_loss.result(),test_accuracy.result()))
            template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, " "Test Accuracy: {}")
            #print(template.format(epoch+1, train_loss,train_accuracy.result()*100, test_loss.result(),test_accuracy.result()*100))

            test_loss.reset_states()
            #train_accuracy.reset_states()
            test_accuracy.reset_states()
            """
            #model.save('model/lm_model_%d'%epoch)
            tf.saved_model.save(model, 'model/lm_model_%d'%epoch)
            tf.saved_model.save(W,'model/W_%d'%epoch)
            end_time = time.time() 
            print("epoch=%d costs time is %d"%(epoch, end_time-begin_time))

 
