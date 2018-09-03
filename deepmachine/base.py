import keras
import tensorflow as tf
import numpy as np

from keras import backend as K
from pathlib import Path
from . import callbacks as cbks
from . import utils
from . import layers
from . import losses
from . import engine


def load_model(filepath, custom_objects={}, compile=False):

    # add DeepMachine Class
    custom_objects.update({'DeepMachine': DeepMachine})

    # add tf for lambda layers
    custom_objects.update({'tf': tf})

    return keras.models.load_model(
        filepath, custom_objects=custom_objects, compile=compile
    )


def _undefined_op(*args, **kwargs):
    raise NotImplementedError


class DeepMachine(keras.Model):
    def __init__(self, network=None, network_config=None, ckpt_path=None, *args, **kwargs):
        # build model from network
        if network:
            if isinstance(network, keras.Model):
                kwargs['inputs'] = network.inputs
                kwargs['outputs'] = network.outputs
            elif callable(network):
                if network_config is not None:
                    inputs = layers.Input(shape=network_config['input_shape'])
                    outputs = network(
                        inputs, **{k: v for k, v in network_config.items() if k != 'input_shape'})

                    kwargs['inputs'] = inputs
                    kwargs['outputs'] = outputs
                else:
                    raise Exception(
                        'Network configuration needed when network is callable')
            elif type(network) is dict:
                kwargs.update(network)
            else:
                raise Exception('Unkown network structure')

        if type(kwargs['inputs']) is not list:
            kwargs['inputs'] = [kwargs['inputs']]

        if type(kwargs['outputs']) is not list:
            kwargs['outputs'] = [kwargs['outputs']]

        self.ckpt_path = ckpt_path

        super().__init__(*args, **kwargs)

    def _prepare_training(self, *args, lr_decay=1, callbacks=[], **kwargs):
        # train_op=train_generator_data_op,
        # valid_data=None,
        # valid_op=None,
        # epochs=None,
        # init_epochs=0,
        # step_per_epoch=None,
        # logdir=None,
        # callbacks=[],
        # verbose=1,

        if lr_decay > 0:
            initial_lr = self.optimizer.lr.eval(K.get_session())
            lr_scheduler = cbks.LearningRateScheduler(
                schedule=lambda epoch: initial_lr * lr_decay ** epoch)
            lr_scheduler.set_model(self)
            callbacks.append(lr_scheduler)

        kwargs['callbacks'] = callbacks

        return args, kwargs

    def fit_generator(self, data_generator, *args, use_multiprocessing=False, workers=4, max_queue_size=256, **kwargs):

        args, kwargs = self._prepare_training(*args, **kwargs)

        # prepare generator
        kwargs['callbacks'].append(data_generator)

        output_generator = utils.enqueue_generator(
            data_generator, use_multiprocessing=use_multiprocessing, workers=workers, max_queue_size=max_queue_size)

        # start training
        history = engine.training.train_monitor(
            self,
            output_generator,
            engine.training.train_generator_data_op,
            *args,
            **kwargs
        )

        return history

    def fit_tf_data(self, tf_dataset, *args, **kwargs):

        args, kwargs = self._prepare_training(*args, **kwargs)

        history = engine.training.train_monitor(
            self,
            tf_dataset,
            engine.training.train_tf_data_op,
            *args,
            **kwargs
        )

        return history
