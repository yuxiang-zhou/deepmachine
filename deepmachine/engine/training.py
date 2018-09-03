import numpy as np
import time
import datetime
import keras

from collections import OrderedDict
from menpo.visualize import print_dynamic
from .. import callbacks as cbks
from keras import backend as K

from ..utils import Summary, channels_to_rgb, max_epoch


def _base_image_summary_op(train_x, train_y, predicts):
    image_summary = {}
    image_summary.update({'inputs/image_%d' % i: imgs
                          for i, imgs in enumerate(train_x)})
    image_summary.update({'target/image_%d' % i: imgs
                          for i, imgs in enumerate(train_y)})
    image_summary.update({'output/image_%d' % i: imgs
                          for i, imgs in enumerate(predicts)})

    return image_summary


def _train_op(model, data, adapter, i_epoch, i_batch, epoch_end, image_summary_ops=[], training_history=[], **kwargs):
    train_x, train_y = adapter(data, i_epoch, i_batch, epoch_end)
    # ----------------------
    #  Train
    # ----------------------
    losses = model.train_on_batch(train_x, train_y)

    if type(losses) is not list:
        losses = [losses]

    # prepare summaries
    scarlar_summary = {
        "losses/total_loss": losses[0]
    }
    scarlar_summary.update(
        {"losses/loss_%d" % i: l for i, l in enumerate(losses[1:])})
    scarlar_summary.update(
        {"learning_rate": model.optimizer.lr.eval(K.get_session())})

    summary = Summary(scarlar_summary)
    
    if epoch_end:
        predicts = model.predict(train_x)

        for img_op in image_summary_ops:
            image_summary = img_op(train_x, train_y, predicts)

            summary.update_images(image_summary)

    return summary


def _valid_op(model, data, adapter, i_epoch, i_batch, epoch_end, image_summary_ops=[], training_history=[], **kwargs):
    valid_x, valid_y = adapter(data, i_epoch, i_batch, epoch_end)
    # ----------------------
    #  Validate
    # ----------------------
    # pred_y = model.predict(valid_x, valid_y)

    # if type(losses) is not list:
    #     losses = [losses]

    # # prepare summaries
    # scarlar_summary = {
    #     "losses/total_loss": losses[0]
    # }
    # scarlar_summary.update(
    #     {"losses/loss_%d" % i: l for i, l in enumerate(losses[1:])})
    # scarlar_summary.update(
    #     {"learning_rate": model.optimizer.lr.eval(K.get_session())})

    # summary = Summary(scarlar_summary)
    
    # if epoch_end:
    #     predicts = model.predict(valid_x)

    #     for img_op in image_summary_ops:
    #         image_summary = img_op(valid_x, valid_y, predicts)

    #         summary.update_images(image_summary)

    # return summary


def train_tf_data_op(model, data, i_epoch, i_batch, epoch_end, **kwargs):
    def adapter(data, i_epoch, i_batch, epoch_end):
        return K.get_session().run(data)

    return _train_op(model, data, adapter, i_epoch, i_batch, epoch_end, **kwargs)


def train_generator_data_op(model, data, i_epoch, i_batch, epoch_end, **kwargs):
    def adapter(data, i_epoch, i_batch, epoch_end):
        return next(data)

    return _train_op(model, data, adapter, i_epoch, i_batch, epoch_end, **kwargs)


def train_monitor(
    models,
    train_data,
    train_op=train_generator_data_op,
    epochs=None,
    init_epochs=0,
    step_per_epoch=None,
    valid_data=None,
    valid_op=None,
    logdir=None,
    restore=True,
    callbacks=[],
    summary_ops=[],
    verbose=1,
):
    # initialise variables
    training_history = cbks.BatchHistory()
    epochs = epochs or np.inf
    step_per_epoch = step_per_epoch or 1000
    total_step = epochs * step_per_epoch
    i_epoch = init_epochs
    callbacks.append(training_history)
    if logdir:
        callbacks.append(cbks.Monitor(logdir, models, restore=restore))
        if restore:
            i_epoch = max_epoch(logdir)

    all_cbks = cbks.CallbackList(callbacks)
    all_summaries = [_base_image_summary_op] + summary_ops
    step_time = []

    # start training
    all_cbks.on_train_begin()
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
                epoch_end=step_per_epoch-i_batch == 1,
                image_summary_ops=all_summaries,
                training_history=training_history
            )

            all_cbks.on_batch_end(i_batch, logs=batch_train_log)
            batch_end_time = time.time()
            step_time.append(batch_end_time - batch_start_time)
            if len(step_time) > step_per_epoch:
                step_time.pop(0)

            avg_step_time = np.mean(step_time)
            estimate_finishing_time = (
                total_step - current_step) * avg_step_time
            batch_train_log.update_scalars({'avg_step_time': avg_step_time,})

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
        if valid_data is not None and valid_op is not None:
            batch_valid_log = valid_op(
                models, 
                train_data,
                image_summary_ops=all_summaries
            )
            logs.update(batch_valid_log)

        if verbose:
            print_dynamic(str(logs))

        # ---------- epoch end -------------
        all_cbks.on_epoch_end(i_epoch, logs=batch_train_log)
        i_epoch += 1

    return training_history
