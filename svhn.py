"""
Builds the svhn network.

Summary of available functions:

# Compute input images and labels for training. If you would like to run
# evaluations, use inputs() instead.
input, labels = distorted_inputs()

# Compute inference on the model inputs to make a prediction.
predictions = inference(inputs)

# Compute the total loss of the predicion with respect to the labels.
loss = loss(predictions, labels)

# Create a graph to run one step of training with respect to the loss.
train_op = train(loss, global_step)

"""

import tensorflow as tf
import os
import re
import tarfile

import svhn_input
from convert_data import convert_to


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', './data/classification/',
                           """Path to the SVHN data directory.""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_boolean('use_sGPU', False,
                            """Train the model using single GPU.""")
tf.app.flags.DEFINE_integer('GPU_id', 0,
                            """Choose which GPU to use, default is 0.""")


# Global constants describing the svhn data set.
IMAGE_SIZE = svhn_input.IMAGE_SIZE
NUM_CLASSES = svhn_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = svhn_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = svhn_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learing rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.


# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
        x: Tensor
    Returns:
        nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/spasity', tf.nn.zero_fraction(x))


def _assign_variable(name, shape, initializer):
    """Helper to create a Variable stored on CPU or GPU memory.

    Args:
        name: name of variable
        shape: list of ints
        initializer: initializer for Variable

    Returns:
        Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32

    if FLAGS.use_sGPU:
        device_id = '/gpu:%d' % FLAGS.GPU_id
    else:
        device_id = '/cpu:0'
    with tf.device(device_id):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """
    Helper to create an initialized Variable with weight decay (e.g. regularization).

    Note that the Variable is initialized with a truncated normal distribution.
    A wight decay is added only if one is specified

    Args:
        name: name of the Variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussion
        wd: add L2 loss weight decay multiplied by this float. If None, weight
            decay is not added for this variable.

    Returns:
        Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _assign_variable(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def basic_inputs(input_method, kwargs):
    """
    Basic input function, used for build distorted_inputs() and inputs()

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.

    Args:
        input_method: Function. Input method to read the data
        kwargs: Dictionary. Keyword args dictionary.

    Raises:
        ValueError: If no data_dir
    """
    with tf.variable_scope('input'):
        images, labels = input_method(**kwargs)

        if FLAGS.use_fp16:
            images = tf.cast(images, tf.float16)
            labels = tf.cast(labels, tf.float16)
        return images, labels


def distorted_inputs():
    """
    Construct distorted input for svhn training using the Reader ops.

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.

    Raises:
        ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    args = {'data_dir': FLAGS.data_dir, 'batch_size': FLAGS.batch_size}
    return basic_inputs(svhn_input.distorted_inputs, args)


def inputs(eval_data):
    """
    Construct input for svhn evaluation using the Reader ops.

    Args:
        eval_data: bool, indicating if one should use the train or eval data set.

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.

    Raises:
        ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    args = {'eval_data': eval_data, 'data_dir': FLAGS.data_dir,
            'batch_size': FLAGS.batch_size}
    return basic_inputs(svhn_input.inputs, args)


def inference(images):
    """
    Build the svhn model.

    Args:
        images: Images returned from distored_inputs() or inputs().

    Returns:
        Logits.
    """
    # We instantiate all variables using tf.get_variable instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().

    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 3, 64],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _assign_variable('biases', [64], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv1)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')

    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 64, 64],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _assign_variable('biases', [64], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv2)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                              stddev=0.04, wd=0.004)
        biases = _assign_variable('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(local3)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                              stddev=0.04, wd=0.004)
        biases = _assign_variable('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        _activation_summary(local4)

    # softmax, i.e. softmax(WX+b)
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                              stddev=1 / 192.0, wd=0.0)
        biases = _assign_variable('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear


def loss(logits, labels):
    """
    Add L2 loss to all the trainable variables.

    Add summary for "Loss" and "Loss/avg".

    Args:
        logits: Logits from inference().
        labels: Labels from distored_inputs() or inputs(). 1-D tensor of shape [batch_size]

    Returns:
        Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    # labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """
    Add summaries for losses in svhn model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
        total_loss: Total loss from loss().
    Returns:
        loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary(l.op.name + ' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step):
    """
    Train svhn model.

    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Args:
        total_loss: Total loss from loss().
        global_step: Integer Variable counting the number of training steps
                     processed.
    Returns:
        train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.scalar_summary('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients descent, and increment global_step by one.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    # Add histogram for gradients
    for grad, var in grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)

    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Add dependencies, make sure that run(train_op) can make
    # apply_gradient_op and variables_averages_op to run.
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def cls_extract():
    """Extract the classification dataset"""
    train_mat_file_path = os.path.join(FLAGS.data_dir, 'train_32x32.mat')
    train_tfr_file_path = os.path.join(FLAGS.data_dir, 'train.tfrecords')
    test_mat_file_path = os.path.join(FLAGS.data_dir, 'test_32x32.mat')
    test_tfr_file_path = os.path.join(FLAGS.data_dir, 'test.tfrecords')
    if not os.path.exists(train_tfr_file_path):
        convert_to(train_mat_file_path, train_tfr_file_path)
    if not os.path.exists(test_tfr_file_path):
        convert_to(test_mat_file_path, test_tfr_file_path)


def det_extract():
    """Extract the detection dataset"""
    FLAGS.data_dir = './data/detection/'
    dest_directory = FLAGS.data_dir
    train_file_path = os.path.join(FLAGS.data_dir, 'train.tar.gz')
    test_file_path = os.path.join(FLAGS.data_dir, 'test.tar.gz')
    if not os.path.exists(os.path.join(dest_directory, 'train')):
        tarfile.open(train_file_path, 'r:gz').extractall(dest_directory)
    if not os.path.exists(os.path.join(dest_directory, 'test')):
        tarfile.open(test_file_path, 'r:gz').extractall(dest_directory)
