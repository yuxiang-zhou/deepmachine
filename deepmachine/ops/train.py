import tensorflow as tf

slim = tf.contrib.slim


def adam(
        initial_learning_rate,
        learning_rate_decay_step,
        batch_size,
        learning_rate_decay_factor,
        moving_average_ckpt):
    # total losses
    total_loss = tf.losses.get_total_loss()

    # learning rate decay
    global_step = slim.get_or_create_global_step()

    learning_rate = tf.train.exponential_decay(
        initial_learning_rate,
        global_step,
        learning_rate_decay_step,
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


def adam(
        initial_learning_rate,
        learning_rate_decay_step,
        batch_size,
        learning_rate_decay_factor,
        moving_average_ckpt):
    # total losses
    total_loss = tf.losses.get_total_loss()

    # learning rate decay
    global_step = slim.get_or_create_global_step()

    learning_rate = tf.train.exponential_decay(
        initial_learning_rate,
        global_step,
        learning_rate_decay_step,
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


def gan(
        initial_learning_rate,
        learning_rate_decay_step,
        batch_size,
        learning_rate_decay_factor,
        moving_average_ckpt):

    # get losses
    gen_loss = tf.reduce_sum(tf.losses.get_losses(
        loss_collection='generator_loss'))
    disc_loss = tf.reduce_sum(tf.losses.get_losses(
        loss_collection='discriminator_loss'))
    
    # learning rate decay
    global_step = slim.get_or_create_global_step()

    learning_rate = tf.train.exponential_decay(
        initial_learning_rate,
        global_step,
        learning_rate_decay_step,
        learning_rate_decay_factor,
        staircase=True)

    tf.summary.scalar('learning_rate', learning_rate)

    # optimiser
    optimizer_gen = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
    optimizer_disc = tf.train.AdamOptimizer(learning_rate, beta1=0.5)

    # train op
    train_variables = tf.trainable_variables()

    gen_variables_to_train = [
        v for v in train_variables if v.name.startswith("generator")]
    gen_train_op = slim.learning.create_train_op(
        gen_loss,
        optimizer_gen,
        variables_to_train=gen_variables_to_train,
        summarize_gradients=True)

    disc_variables_to_train = [
        v for v in train_variables if v.name.startswith("discriminator")]
    disc_train_op = slim.learning.create_train_op(
        disc_loss,
        optimizer_disc,
        variables_to_train=disc_variables_to_train,
        summarize_gradients=True)

    train_op = [gen_train_op, disc_train_op]

    return train_op


def cyclegan(
        initial_learning_rate,
        learning_rate_decay_step,
        batch_size,
        learning_rate_decay_factor,
        moving_average_ckpt):

    # get losses
    generator_loss_AB = tf.reduce_sum(tf.losses.get_losses(
        loss_collection='generator_loss_AB'))
    generator_loss_BA = tf.reduce_sum(tf.losses.get_losses(
        loss_collection='generator_loss_BA'))
    disc_loss_A = tf.reduce_sum(tf.losses.get_losses(
        loss_collection='discriminator_loss_A'))
    disc_loss_B = tf.reduce_sum(tf.losses.get_losses(
        loss_collection='discriminator_loss_B'))

    losses_all = [generator_loss_AB, generator_loss_BA, disc_loss_A, disc_loss_B]
    variables_prefixes = ['generatorAB', 'generatorBA', 'discriminatorA', 'discriminatorB']

    # learning rate decay
    global_step = slim.get_or_create_global_step()

    learning_rate = tf.train.exponential_decay(
        initial_learning_rate,
        global_step,
        learning_rate_decay_step * len(variables_prefixes),
        learning_rate_decay_factor,
        staircase=True)

    tf.summary.scalar('learning_rate', learning_rate)

    # train op
    train_variables = tf.trainable_variables()

    train_op = []
    for loss, vname in zip(losses_all, variables_prefixes):
        variables_to_train = [v for v in train_variables if v.name.startswith(vname)]
        print('# of %s variables:%d'%(vname, len(variables_to_train)))
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
        t_op = slim.learning.create_train_op(
            loss,
            optimizer,
            variables_to_train=variables_to_train,
            summarize_gradients=True)

        train_op.append(t_op)
            
    return train_op


def adam_ae(
        initial_learning_rate,
        learning_rate_decay_step,
        batch_size,
        learning_rate_decay_factor,
        moving_average_ckpt):

    # get losses
    reg_loss = tf.reduce_sum(tf.losses.get_losses(
        loss_collection='regression_loss'))
    rec_loss = tf.reduce_sum(tf.losses.get_losses(
        loss_collection='reconstruction_loss'))
    
    # learning rate decay
    global_step = slim.get_or_create_global_step()

    learning_rate = tf.train.exponential_decay(
        initial_learning_rate,
        global_step,
        learning_rate_decay_step,
        learning_rate_decay_factor,
        staircase=True)

    tf.summary.scalar('learning_rate', learning_rate)

    # optimiser
    optimizer_reg = tf.train.AdamOptimizer(learning_rate)
    optimizer_rec = tf.train.AdamOptimizer(learning_rate)

    # train op
    train_variables = tf.trainable_variables()

    reg_variables_to_train = [
        v for v in train_variables if v.name.startswith("regression")]
    reg_train_op = slim.learning.create_train_op(
        reg_loss,
        optimizer_reg,
        variables_to_train=reg_variables_to_train,
        summarize_gradients=True)

    rec_variables_to_train = [
        v for v in train_variables if v.name.startswith("reconstruction")]
    rec_train_op = slim.learning.create_train_op(
        rec_loss,
        optimizer_rec,
        variables_to_train=rec_variables_to_train,
        summarize_gradients=True)

    train_op = [reg_train_op, rec_train_op]

    return train_op