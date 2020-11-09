import tensorflow as tf
import sys
file_name = sys.argv[1]
writer = tf.io.TFRecordWriter(file_name)
f = sys.stdin
for data in f:
    splits = data.split ('\t')
    x , y = splits
    x_list = x.split(",")
    x_int_list = list(map(int, x_list))
    y_int = int(y)
    example = tf.train.Example(features=tf.train.Features(feature={
                        "x": tf.train.Feature(int64_list=tf.train.Int64List(value=x_int_list)),
                        "y": tf.train.Feature(int64_list=tf.train.Int64List(value=[y_int]))
                }))
    writer.write(example.SerializeToString())
writer.close()
