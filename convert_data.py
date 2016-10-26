import tensorflow as tf
import numpy as np
from scipy.io import loadmat


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(mat_filepath, tfr_filepath):
    """
    Convert svhn dataset from matlab format to TFRecord format

    Args:
        mat_filepath: String. File path to matlab data file
        tfr_filepath: String. File path to TFRecord data file
    """
    mat_data = loadmat(mat_filepath)
    images = mat_data['X'].transpose([3, 0, 1, 2])
    labels = mat_data['y']
    num_examples = len(labels)

    # rows = images.shape[1]
    # cols = images.shape[2]
    # depth = images.shape[3]

    print('Writing ', tfr_filepath)
    writer = tf.python_io.TFRecordWriter(tfr_filepath)
    for idx in range(num_examples):
        image_raw = images[idx].tostring()
        label = int(labels[idx])
        if label == 10:
            label = 0
        example = tf.train.Example(features=tf.train.Features(feature={
            # 'height': _int64_feature(rows),
            # 'width': _int64_feature(cols),
            # 'depth': _int64_feature(depth),
            'label': _int64_feature(label),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()
    del mat_data
