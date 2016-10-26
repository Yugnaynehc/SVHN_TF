import os

import tensorflow as tf

# Use for crop the images
# Process images of this size. Note that this differs from the original svhn
# image size of 32x32. If one alters this number, then the entire model
# architecture will fhange and any model would need to be retrained.
IMAGE_SIZE = 24

# Global constants describing the svhn data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 73257
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 26032


def read_svhn(filename_queue):
    """
    Read svhn data from TFRecord

    Args:
        filename_queue: String list containing data filenames

    Returns:
        image: 3-D Tensor of [height, width, 3] of type float32
        label: 1-D Tensor of type int32
    """
    # Dimensions of the images in the svhn dataset.
    # label_bytes = 1
    height = 32
    width = 32
    depth = 3
    # image_bytes = result.height * result.width * result.depth
    # record_bytes = label_bytes + image_bytes
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'image_raw': tf.FixedLenFeature([], tf.string)
                                       })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [height, width, depth])
    image = tf.cast(image, tf.float32)
    label = tf.cast(features['label'], tf.int32)

    return image, label


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """
    Construct a queued batch of images and labels.

    Args:
        image: 3-D Tensor of [height, width, 3] of type.float32.
        label: 1-D Tensor of type.int32
        min_queue_examples: int32, minimum number of samples to retain
                     in the queue that provides of batches of examples
        batch_size: Number of images per batch.
        shuffle: boolean indicating whether to use a shuffling queue.

    Returns:
        images: Images. 4D tensor of [batch_size, height, width, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' iamges+labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.image_summary('images', images)

    # return images, tf.reshape(label_batch, [batch_size])
    return images, label_batch


def distorted_inputs(data_dir, batch_size):
    """
    Construct distorted input for svhn training using the Reader ops.

    Args:
        data_dir: Path to the svhn data directory.
        batch_size: Number of images per batch.

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    filenames = [os.path.join(data_dir, 'train.tfrecords')]

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    image, label = read_svhn(filename_queue)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for training the network. Note the many random
    # distortions applied to hte image.

    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(image, [height, width, 3])

    # Randomly file the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)

    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(distorted_image)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print('Filling queue with %d svhn images before starting to train. '
          'This will take a minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, label,
                                           min_queue_examples, batch_size,
                                           shuffle=True)


def inputs(eval_data, data_dir, batch_size):
    """
    Construct input for svhn evaluation using the Reader ops.

    Args:
        eval_data: bool, indicating if one should use the train or eval data set.
        data_dir: Path to the svhn data directory.
        batch_size: Number of images per batch.

    Returns:
        images: Images. 4-D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1-D tensor of [batch_size] size.
    """
    if eval_data:
        filenames = [os.path.join(data_dir, 'test.tfrecords')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
    else:
        filenames = [os.path.join(data_dir, 'train.tfrecords')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    image, label = read_svhn(filename_queue)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for evaluation.
    # Crop the central [heigh, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(image, width, height)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(resized_image)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)
    print('Filling queue with %d svhn images before starting to eval. '
          'This will take a minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, label,
                                           min_queue_examples, batch_size, shuffle=False)
