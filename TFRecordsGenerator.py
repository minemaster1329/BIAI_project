import tensorflow as tf
from glob import glob
import os
import random


def serialize_example(image, label, label_index):
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.encode('utf-8')])),
        'label_index': tf.train.Feature(int64_list=tf.train.Int64List(value=[label_index]))
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def make_tfrecords(path, record_file='/media/dane2/tfrecords/images.tfrecords'):
    classes = os.listdir(path)
    with tf.io.TFRecordWriter(record_file) as writer:
        files_list = glob(path + '/*/*')
        random.shuffle(files_list)
        for filename in files_list:
            image_string = open(filename, 'rb').read()
            category = filename.split('/')[-2]
            label = classes.index(category)
            tf_example = serialize_example(image_string, category, label)
            writer.write(tf_example)


make_tfrecords('/media/dane2/BIAI_datasets/main/CropDisease/Crop___DIsease')
print("Done!")
