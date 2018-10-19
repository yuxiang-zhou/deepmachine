import keras
import tensorflow as tf
import numpy as np
import menpo.io as mio

from menpo.image import Image
from menpo.shape import PointCloud
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
    def __init__(self, network=None, network_config=None, *args, **kwargs):
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

        super().__init__(*args, **kwargs)

    def _config_sequence_data(
        self,
        sequence_data,
        is_training=True,
        use_multiprocessing=True,
        workers=4,
        max_queue_size=256,
        **kwargs
    ):
        queue_data = utils.enqueue_generator(
            sequence_data,
            use_multiprocessing=use_multiprocessing,
            workers=workers,
            max_queue_size=max_queue_size)

        return queue_data, kwargs

    def _prepare_training(self, train_data, lr_decay=1, callbacks=[], valid_data=None, **kwargs):
        # train_data,
        # train_op=train_generator_data_op,
        # epochs=None,
        # init_epochs=0,
        # step_per_epoch=None,
        # valid_data=None,
        # valid_op=None,
        # valid_steps=None,
        # logdir=None,
        # restore=True,
        # callbacks=[],
        # summary_ops=[],
        # verbose=1,

        if isinstance(train_data, utils.Sequence):
            kwargs['train_op'] = kwargs['train_op'] or engine.training.train_generator_data_op
            kwargs['step_per_epoch'] = kwargs['step_per_epoch'] or len(train_data)
            callbacks.append(train_data)
            train_data, kwargs = self._config_sequence_data(
                train_data, **kwargs)
        else:
            kwargs['train_op'] = kwargs['train_op'] or engine.training.train_tf_data_op

        
        if valid_data:
            if isinstance(valid_data , utils.Sequence):
                kwargs['valid_op'] = kwargs['valid_op'] or engine.training.valid_generator_data_op
                kwargs['valid_steps'] = kwargs['valid_steps'] or len(valid_data)
                callbacks.append(valid_data)
                valid_data, kwargs = self._config_sequence_data(valid_data, is_training=True, **kwargs)
            else:
                kwargs['valid_op'] = kwargs['valid_op'] or engine.training.valid_tf_data_op
                kwargs['valid_steps'] = kwargs['valid_steps'] or 10

            kwargs['valid_data'] = valid_data

        if lr_decay > 0:
            initial_lr = self.optimizer.lr.eval(K.get_session())
            lr_scheduler = cbks.LearningRateScheduler(
                schedule=lambda epoch: initial_lr * lr_decay ** epoch)
            lr_scheduler.set_model(self)
            callbacks.append(lr_scheduler)

        kwargs['callbacks'] = callbacks

        return train_data, kwargs

    def fit(self,
            train_data,
            train_op=None,
            epochs=None,
            init_epochs=0,
            step_per_epoch=None,
            valid_data=None,
            valid_op=None,
            valid_steps=None,
            logdir=None,
            restore=True,
            lr_decay=1,
            callbacks=[],
            summary_ops=[],
            verbose=1,
            **kwargs
            ):

        data, kwargs = self._prepare_training(
            train_data,
            train_op=train_op,
            epochs=epochs,
            init_epochs=init_epochs,
            step_per_epoch=step_per_epoch,
            valid_data=valid_data,
            valid_op=valid_op,
            valid_steps=valid_steps,
            logdir=logdir,
            restore=restore,
            lr_decay=lr_decay,
            callbacks=callbacks,
            summary_ops=summary_ops,
            verbose=verbose,
            **kwargs
        )

        history = engine.training.train_monitor(self, data, **kwargs)

        return history