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
K = tf.keras.backend


def load_model(filepath, custom_objects={}, compile=False):

    # add DeepMachine Class
    custom_objects.update({'DeepMachine': DeepMachine})

    # add tf for lambda layers
    custom_objects.update({'tf': tf})

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

            if type(kwargs['inputs']) is not list:
                kwargs['inputs'] = [kwargs['inputs']]

            if type(kwargs['outputs']) is not list:
                kwargs['outputs'] = [kwargs['outputs']]

        self.ckpt_path = ckpt_path
        if ckpt_path:
            self.default_cbs = [
                callbacks.TBSummary(
                    log_dir=self.ckpt_path,
                    write_grads=True,
                    histogram_freq=1
                ),
                callbacks.ModelCheckpoint(
                    self.ckpt_path + '/weights.{epoch:05d}.hdf5'),
            ]

        super().__init__(*args, **kwargs)

    def _fitting_args_check(self, *args, summaries=None, **kwargs):
        # appending default callbacks
        if 'callbacks' not in kwargs or kwargs['callbacks'] is None:
            kwargs['callbacks'] = []
        kwargs['callbacks'] += self.default_cbs

        self.progress_bar = None
        self.epoch = 0
        self.summaries = summaries

        if len(args) > 0 and isinstance(args[0], tf.keras.utils.Sequence):
            kwargs['steps_per_epoch'] = len(args[0])

        self.steps_per_epoch = kwargs['steps_per_epoch'] or 1

        if 'epochs' in kwargs:
            self.progress_bar = utils.Progbar(target=kwargs['epochs'] * self.steps_per_epoch)
            kwargs['verbose'] = 0


        return args, kwargs

    def compile(self, optimizer, lr_decay=1.0, **kwargs):

        initial_lr = tf.keras.backend.get_session().run(optimizer.lr)

        if hasattr(self, 'default_cbs') and self.default_cbs:
            self.default_cbs.append(callbacks.LearningRateScheduler(schedule=lambda epoch: initial_lr * (lr_decay ** epoch)))

        super().compile(optimizer, **kwargs)

    def fit_generator(self, *args, **kwargs):

        args, kwargs = self._fitting_args_check(*args, **kwargs)

        trained_model = super().fit_generator(*args, **kwargs)

        return trained_model

    def fit(self, *args, **kwargs):

        args, kwargs = self._fitting_args_check(*args, **kwargs)

        # start training
        training_using_queue_runner = len(
            tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)) > 0
        # Fit the model using data from the TFRecord data tensors. If queue runner exist.
        if training_using_queue_runner:
            sess = K.get_session()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)

        # training
        trained_model = super().fit(*args, **kwargs)

        # Clean up the TF session.
        if training_using_queue_runner:
            coord.request_stop()
            coord.join(threads)

        return trained_model
