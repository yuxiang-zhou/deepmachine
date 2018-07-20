import tensorflow as tf
from pathlib import Path
from . import callbacks
from . import utils
from . import layers
from . import losses

model_from_config = tf.keras.models.model_from_config
model_from_json = tf.keras.models.model_from_json
model_from_yaml = tf.keras.models.model_from_yaml
save_model = tf.keras.models.save_model


def load_model(filepath, custom_objects={}, compile=True):

    # add DeepMachine Class
    custom_objects.update({'DeepMachine': DeepMachine})

    # add tf for lambda layers
    custom_objects.update({'tf': tf})

    # add custom losses
    custom_objects.update(
        {loss: getattr(losses, loss)
         for loss in dir(losses) if loss.startswith('loss_')}
    )

    return tf.keras.models.load_model(
        filepath, custom_objects=custom_objects, compile=compile
    )


def _undefined_op(*args, **kwargs):
    raise NotImplementedError


class DeepMachine(tf.keras.Model):
    def __init__(self, network=None, network_config=None, ckpt_path=None, *args, **kwargs):
        if network is not None:
            if isinstance(network, tf.keras.Model):
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

        self.ckpt_path = ckpt_path
        if ckpt_path:
            self.default_cbs = [
                callbacks.TBSummary(log_dir=self.ckpt_path,
                                    write_grads=True, histogram_freq=1),
                callbacks.ModelCheckpoint(
                    self.ckpt_path + '/weights.{epoch:05d}.hdf5'),
            ]
        super().__init__(*args, **kwargs)

    def fit_generator(self, *args, **kwargs):
        if 'callbacks' not in kwargs or kwargs['callbacks'] is None:
            kwargs['callbacks'] = []
        kwargs['callbacks'] += self.default_cbs

        self.progress_bar = None if 'epochs' not in kwargs else utils.Progbar(
            target=kwargs['epochs'])
        return super().fit_generator(*args, **kwargs)
