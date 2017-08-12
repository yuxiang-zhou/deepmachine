import tensorflow as tf
from deepmachine.flags import FLAGS

slim = tf.contrib.slim


def adam(
        initial_learning_rate=FLAGS.initial_learning_rate,
        learning_rate_decay_step=FLAGS.learning_rate_decay_step,
        batch_size=FLAGS.batch_size,
        learning_rate_decay_factor=FLAGS.learning_rate_decay_factor,
        moving_average_ckpt=FLAGS.moving_average_ckpt):
    # total losses
    total_loss = tf.losses.get_total_loss()

    # learning rate decay
    global_step = slim.get_or_create_global_step()

    learning_rate = tf.train.exponential_decay(
        initial_learning_rate,
        global_step,
        learning_rate_decay_step / batch_size,
        learning_rate_decay_factor,
        staircase=True)

    tf.summary.scalar('learning_rate', learning_rate)

    # train op
    optimizer = tf.train.AdamOptimizer(learning_rate)

    if moving_average_ckpt:
        optimizer = tf.contrib.opt.MovingAverageOptimizer(optimizer)

    train_op = slim.learning.create_train_op(
        total_loss,
        optimizer,
        summarize_gradients=True)

    return train_op
