import tensorflow as tf
import numpy as np
import time
from datetime import datetime
import os.path

import svhn

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('cls', True,
                            """Determine whether target is classification or not."""
                            """ If True, we solve a classification problem,"""
                            """ else a detection problem.""")
tf.app.flags.DEFINE_string('train_log_dir', './log/classification/train',
                           """Directory where to write training event logs and checkpoint""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('max_steps', 10000,
                            """Number of batches to run.""")


def train():
    """Train svhn for a number of steps."""
    with tf.Graph().as_default():

        global_step = tf.Variable(0, trainable=False)

        # Get (distorted) images and labels for svhn.
        images, labels = svhn.distorted_inputs()

        # Build a graph that computes the logits predictions
        # from the inference model.
        logits = svhn.inference(images)

        # Calculate loss.
        loss = svhn.loss(logits, labels)

        # Build a graph that trains the model with one batch of examples
        # and updates the model parameters.
        train_op = svhn.train(loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Build an variables initialization operation
        init = tf.initialize_all_variables()

        # Start running operations on the graph.
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.train.SummaryWriter(FLAGS.train_log_dir, sess.graph)

        for step in range(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                # Print the loss and tra.ining speed
                examples_per_sec = FLAGS.batch_size / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss=%.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))
            if step % 100 == 0:
                # Write the summaries
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                # Save the model checkpoint
                checkpoint_path = os.path.join(FLAGS.train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):
    if FLAGS.cls is True:
        # Solve a classification problem
        svhn.cls_extract()
    else:
        # Solve a detection
        FLAGS.train_log_dir = './log/detection/train'
        svhn.det_extract()
    if tf.gfile.Exists(FLAGS.train_log_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_log_dir)
    tf.gfile.MakeDirs(FLAGS.train_log_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
