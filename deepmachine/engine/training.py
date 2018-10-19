import numpy as np
import time
import datetime
import keras
import tensorflow as tf

from collections import OrderedDict
from functools import partial
from menpo.visualize import print_dynamic
from .. import callbacks as cbks
from keras import backend as K

from ..utils import Summary, channels_to_rgb, max_epoch

def generator_adapter(data, *args, **kwargs):
    return next(data)


def tf_dataset_adapter(data, *args, **kwargs):
    return K.get_session().run(data)


def identity_adapter(data, *args, **kwargs):
    return data


def _base_image_summary_op(train_x, train_y, predicts):
    train_x = [] if train_x is None else train_x
    train_y = [] if train_y is None else train_y
    predicts = [] if predicts is None else predicts

    image_summary = {}
    image_summary.update({'inputs/image_%d' % i: imgs
                          for i, imgs in enumerate(train_x) if len(imgs.shape) == 4})
    image_summary.update({'target/image_%d' % i: imgs
                          for i, imgs in enumerate(train_y) if len(imgs.shape) == 4})
    image_summary.update({'output/image_%d' % i: imgs
                          for i, imgs in enumerate(predicts) if len(imgs.shape) == 4})

    return image_summary


def _train_op(model, data, i_epoch, i_batch, epoch_end, adapter=identity_adapter, image_summary_ops=[], training_history=[], **kwargs):
    train_x, train_y = adapter(data, i_epoch, i_batch, epoch_end)
    # ----------------------
    #  Train
    # ----------------------
    losses = model.train_on_batch(train_x, train_y)
    if type(losses) is not list and not isinstance(losses, np.ndarray):
        losses = [losses]

    # prepare summaries
    scarlar_summary = {
        "losses/total_loss": losses[0]
    }
    scarlar_summary.update(
        {"losses/loss_%d" % i: l for i, l in enumerate(losses[1:])})
    scarlar_summary.update(
        {"status/learning_rate": model.optimizer.lr.eval(K.get_session())})

    summary = Summary(scarlar_summary)
    
    if epoch_end:
        predicts = model.predict(train_x)
        if len(predicts) != len(train_y):
            predicts = [predicts]
        for img_op in image_summary_ops:
            image_summary = img_op(train_x, train_y, predicts)

            summary.update_images(image_summary)

    return summary


def _valid_op(model, data, i_epoch, i_batch, epoch_end, adapter=identity_adapter, image_summary_ops=[], training_history=[], **kwargs):
    valid_x, valid_y = adapter(data, i_epoch, i_batch, epoch_end)
    # ----------------------
    #  Validate
    # ----------------------
    losses = model.evaluate(valid_x, valid_y, verbose=0)

    if type(losses) is not list and not isinstance(losses, np.ndarray):
        losses = [losses]

    # prepare summaries
    scarlar_summary = {
        "losses/total_loss": losses[0]
    }
    scarlar_summary.update(
        {"losses/loss_%d" % i: l for i, l in enumerate(losses[1:])})

    summary = Summary(scarlar_summary)
    
    if epoch_end:
        predicts = model.predict(valid_x)

        for img_op in image_summary_ops:
            image_summary = img_op(valid_x, valid_y, predicts)

            summary.update_images(image_summary)

    return summary

train_tf_data_op = partial(_train_op, adapter=tf_dataset_adapter)
train_generator_data_op = partial(_train_op, adapter=generator_adapter)
valid_tf_data_op = partial(_valid_op, adapter=tf_dataset_adapter)
valid_generator_data_op = partial(_valid_op, adapter=generator_adapter)


def train_monitor(
    models,
    train_data,
    train_op=train_generator_data_op,
    epochs=None,
    init_epochs=0,
    step_per_epoch=None,
    valid_data=None,
    valid_op=None,
    valid_steps=None,
    logdir=None,
    restore=True,
    callbacks=[],
    summary_ops=[],
    verbose=1,
):
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=K.get_session())
    # initialise variables
    training_history = cbks.BatchHistory()
    monitor = cbks.Monitor(logdir, models, restore=restore)
    epochs = epochs or np.inf
    step_per_epoch = step_per_epoch or 1000
    total_step = epochs * step_per_epoch
    step_time = [0]

    i_epoch = init_epochs
    if restore and logdir:
        i_epoch = max_epoch(logdir) + 1
    callbacks.append(training_history)
    callbacks.append(monitor)
    all_cbks = cbks.CallbackList(callbacks)
    all_summaries = [_base_image_summary_op] + summary_ops

    # start training
    all_cbks.on_train_begin()
    print("Training started from epoch: %d"%i_epoch)
    while i_epoch < epochs:
        all_cbks.on_epoch_begin(i_epoch)

        # ---------- epoch start -----------
        for i_batch in range(step_per_epoch):
            all_cbks.on_batch_begin(i_batch)
            current_step = i_epoch * step_per_epoch + i_batch

            # ---------- batch start -----------
            batch_start_time = time.time()
            batch_train_log = train_op(
                models, 
                train_data, 
                i_epoch=i_epoch, 
                i_batch=i_batch, 
                epoch_end=i_batch + 1 == step_per_epoch,
                image_summary_ops=all_summaries,
                training_history=training_history
            )

            # summary preparation
            avg_step_time = np.mean(step_time)
            estimate_finishing_time = (
                total_step - current_step) * avg_step_time
            batch_train_log.update_scalars({'status/avg_step_time': avg_step_time,})
            all_cbks.on_batch_end(i_batch, logs=batch_train_log)

            # build batch summary
            logs = Summary({
                'epoch': '%.0f/%.0f' % (i_epoch, epochs),
                'batch': '%.0f/%.0f' % (i_batch, step_per_epoch),
                'remaining_time': str(datetime.timedelta(seconds=estimate_finishing_time)),
            })
            logs.update(batch_train_log)
            if verbose > 1:
                print(str(logs))
            # ---------- batch end -------------
            batch_end_time = time.time()
            step_time.append(batch_end_time - batch_start_time)
            if len(step_time) > np.min([10, step_per_epoch]):
                step_time.pop(0)
        
        # Run Validation if Provided
        if valid_data is not None and valid_op is not None and valid_steps is not None:
            
            for valid_i_step in range(valid_steps):
                batch_valid_log = valid_op(
                    models, 
                    valid_data,
                    i_epoch=valid_i_step,
                    i_batch=None,
                    epoch_end=valid_i_step+1 == valid_steps,
                    image_summary_ops=all_summaries
                )
                monitor.on_valid_batch(valid_i_step, logs=batch_valid_log)
                
            batch_train_log.update_images(batch_valid_log.images)

        if verbose:
            print_dynamic(str(logs))

        # ---------- epoch end -------------
        all_cbks.on_epoch_end(i_epoch, logs=batch_train_log)
        i_epoch += 1

    coord.request_stop()
    coord.join(threads)

    return training_history
