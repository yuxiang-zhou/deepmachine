# basic library
from deepmachine.utils.machine import multi_gpu_model
import deepmachine as dm
import os
import shutil
import math
import time
import h5py
import sklearn
import menpo.io as mio
import menpo3d.io as m3io
import numpy as np
import datetime
from pathlib import Path
from functools import partial

# deepmachine
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)


def format_folder(FLAGS):
    feature_string = ''
    if FLAGS.uv:
        feature_string += '_uv'

    if FLAGS.coord:
        feature_string += '_xy'

    if FLAGS.heatmap:
        feature_string += '_hm'

    post_fix = 'lr{:.5f}_d{:.3f}_b{:03d}_opt-{}_loss-{}{}'.format(
        FLAGS.lr, FLAGS.lr_decay, FLAGS.batch_size, FLAGS.opt, FLAGS.loss_type, feature_string
    )

    logdir = FLAGS.logdir if 'model_' in FLAGS.logdir else "{}/model_{}".format(
        FLAGS.logdir, post_fix
    )

    return logdir


class ArcDense(tf.keras.layers.Layer):

    def __init__(
            self,
            units,
            **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_variable(
            'kernel', shape=(input_dim, self.units))

    def call(self, inputs):
        # l2 normalize parameters
        norm_x = tf.nn.l2_normalize(inputs, axis=-1)
        norm_w = tf.nn.l2_normalize(self.kernel, axis=0)

        # compute arc distance
        output = tf.tensordot(norm_x, norm_w, axes=[[1], [0]])
        return output


def ResiduleModule(x, out_channels, ks=3, s=1, activation=tf.nn.relu, **kwargs):
    in_channels = x.get_shape().as_list()[-1]

    # conv
    y = tf.layers.BatchNormalization()(x)
    y = tf.layers.conv2d(y, out_channels, ks, strides=1,
                         padding='same', activation=activation)
    y = tf.layers.BatchNormalization()(y)
    y = tf.layers.conv2d(y, out_channels, ks, strides=s,
                         padding='same', activation=activation)
    y = tf.layers.BatchNormalization()(y)

    # residule
    if in_channels != out_channels or s > 1:
        x = tf.layers.conv2d(x, out_channels, 1, strides=s,
                             padding='same', activation=None)

    return y + x


def ArcFace(inputs, embeding, nf=64, n_classes=None, dropout=0.3, loss_type='arc', **kwargs):

    net = tf.layers.BatchNormalization()(inputs)
    # input shape: 112 * 112 * c
    net = tf.layers.conv2d(net, nf, 3, strides=1,
                           padding='same', activation=tf.nn.relu)
    # shape: 112 * 112 * 64
    net = ResiduleModule(net, nf, s=2, **kwargs)
    # shape: 56 * 56 * 64
    net = ResiduleModule(net, nf, s=1, **kwargs)
    # shape: 56 * 56 * 64
    net = ResiduleModule(net, nf, s=1, **kwargs)
    # shape: 56 * 56 * 64
    net = ResiduleModule(net, nf*2, s=2, **kwargs)
    # shape: 28 * 28 * 128
    for _ in range(12):
        net = ResiduleModule(net, nf*2, s=1, **kwargs)
        # shape: 28 * 28 * 128
    net = ResiduleModule(net, nf*4, s=2, **kwargs)
    # shape: 14 * 14 * 256
    for _ in range(29):
        net = ResiduleModule(net, nf*4, s=1, **kwargs)
        # shape: 14 * 14 * 256
    net = ResiduleModule(net, nf*8, s=2, **kwargs)
    # shape: 7 * 7 * 512
    for _ in range(2):
        net = ResiduleModule(net, nf*8, s=1, **kwargs)
        # shape: 7 * 7 * 512
    net = tf.layers.BatchNormalization()(net)
    net = tf.layers.Dropout(dropout)(net)
    net = tf.layers.Flatten()(net)
    embeding_output = tf.layers.Dense(
        embeding, activation=None, name='feature_embeddings')(net)

    if loss_type == 'arc':
        # arcface
        logits = tf.nn.relu(embeding_output)
        logits = ArcDense(n_classes)(logits)
    else:
        # softmax
        logits = tf.nn.relu(embeding_output)
        logits = tf.layers.Dense(n_classes, activation=None)(logits)

    return logits, embeding_output


# Dataset
def get_data_fn(FLAGS, N_CLASSES, INPUT_SHAPE=112):
    NUM_GPUS = len(FLAGS.gpu.split(','))
    BATCH_SIZE = FLAGS.batch_size

    def data_fn():

        keys_to_features = dm.utils.union_dict([
            dm.data.provider.features.image_feature(),
            dm.data.provider.features.tensor_feature('uv'),
            dm.data.provider.features.array_feature('label'),
            dm.data.provider.features.lms_feature('landmarks'),
        ])

        dataset = tf.data.TFRecordDataset(
            FLAGS.dataset_path, num_parallel_reads=FLAGS.no_thread)

        # Shuffle the dataset
        dataset = dataset.shuffle(
            buffer_size=BATCH_SIZE * NUM_GPUS * FLAGS.no_thread)

        # Repeat the input indefinitly
        dataset = dataset.repeat()

        # Generate batches
        dataset = dataset.batch(BATCH_SIZE)

        # example proto decode
        def _parse_function(example_proto):

            parsed_features = tf.parse_example(example_proto, keys_to_features)
            feature_dict = {}

            # parse image
            def parse_single_image(feature):

                m = tf.image.decode_jpeg(feature, channels=3)
                m = tf.reshape(m, [INPUT_SHAPE, INPUT_SHAPE, 3])
                m = tf.to_float(m) / 255.
                return m

            # parse image
            feature_dict['image'] = tf.map_fn(
                parse_single_image, parsed_features['image'], dtype=tf.float32)

            # parse label
            m = tf.decode_raw(parsed_features['label'], tf.float32)
            m = tf.reshape(m, [-1, 1])
            m = tf.cast(m, tf.int64)
            m = tf.contrib.layers.one_hot_encoding(m, N_CLASSES)
            m = tf.squeeze(m)
            feature_dict['label'] = m

            # parse uv
            if FLAGS.uv:
                m = tf.decode_raw(parsed_features['uv'], tf.float32)
                m = tf.reshape(m, [-1, INPUT_SHAPE, INPUT_SHAPE, 2])
                feature_dict['uv'] = m

            # parse heatmap
            if FLAGS.heatmap:
                def parse_single_hm(feature, n_landmarks=5):
                    # load features
                    gt_lms = tf.decode_raw(feature, tf.float32)
                    visible = list(range(n_landmarks))
                    image_height = INPUT_SHAPE
                    image_width = INPUT_SHAPE

                    # formation
                    gt_lms = tf.reshape(gt_lms, (n_landmarks, 2))
                    gt_heatmap = dm.utils.tf_lms_to_heatmap(
                        gt_lms, image_height, image_width, n_landmarks, visible)
                    gt_heatmap = tf.transpose(gt_heatmap, perm=[1, 2, 0])

                    return gt_heatmap

                m = tf.map_fn(parse_single_hm,
                              parsed_features['landmarks'], dtype=tf.float32)
                feature_dict['heatmap'] = m

            return feature_dict, feature_dict['label']

        # Parse the record into tensors.
        dataset = dataset.map(
            _parse_function, num_parallel_calls=FLAGS.no_thread)

        return dataset

    return data_fn


# Model
def model_optimizer_fn(FLAGS, N_CLASSES, NF, EPOCH_STEPS):
    LR = FLAGS.lr

    # losses Layer
    def arc_loss(gt_label, emb, s=64., m1=1., m2=0.3, m3=0.):
        # arc feature
        arc = tf.boolean_mask(emb, gt_label > 0)
        t_arc = tf.acos(arc)
        t_arc = tf.cos(t_arc * m1 + m2) - m3

        # update embedding
        diff = t_arc - arc
        diff = gt_label * diff[:, None]
        new_emb = diff + emb

        # scale embedding
        new_emb *= s

        # simplified version
        # new_emb = emb - gt_label * 0.25
        # new_emb = new_emb * s

        return tf.losses.softmax_cross_entropy(gt_label, new_emb)

    def model_fn(
            features,  # This is batch_features from input_fn
            labels,   # This is batch_labels from input_fn
            mode,     # An instance of tf.estimator.ModeKeys
            params):  # Additional configuration:

        if mode != tf.estimator.ModeKeys.PREDICT:
            labels.set_shape([None, N_CLASSES])

        tf.summary.image('input/image', features['image'])

        inputs = features['image']

        if FLAGS.uv:
            inputs = tf.concat([inputs, features['uv']], axis=-1)
            tf.summary.image(
                'input/uv', tf.concat([tf.to_float(features['uv'][..., :-1] > 0), features['uv']], axis=-1))

        if FLAGS.coord:
            _, h, w, _ = inputs.get_shape().as_list()
            rhm = tf.tile(tf.range(h)[..., None], [1, w])
            rwm = tf.tile(tf.range(w)[None, ...], [h, 1])
            rhm_norm = tf.to_float(rhm) / w
            rwm_norm = tf.to_float(rwm) / h
            inputs = tf.concat([
                inputs,
                rhm_norm[None, ..., None] * tf.ones_like(inputs),
                rwm_norm[None, ..., None] * tf.ones_like(inputs)
            ], axis=-1)

        if FLAGS.heatmap:
            inputs = tf.concat([inputs, features['heatmap']], axis=-1)
            tf.summary.image(
                'input/heatmap', dm.utils.tf_n_channel_rgb(features['heatmap'], 5))

        logits, embedding = ArcFace(
            inputs, 512, nf=NF, n_classes=N_CLASSES, loss_type=FLAGS.loss_type)

        # Build estimator spec
        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
            'embedding': embedding
        }

        # PREDICT mode
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
        if FLAGS.loss_type == 'arc':
            loss = arc_loss(labels, logits)
        else:
            loss = tf.losses.softmax_cross_entropy(labels, logits)
        global_steps = tf.train.get_global_step()
        tf.summary.scalar('loss', loss)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:

            learning_rate = tf.train.piecewise_constant(
                global_steps,
                [EPOCH_STEPS * (FLAGS.n_epoch // 2), EPOCH_STEPS * int(FLAGS.n_epoch * 0.7), EPOCH_STEPS * int(FLAGS.n_epoch * 0.8)],
                [LR, LR/10., LR/100., LR/1000.]
            )

            tf.summary.scalar('lr', learning_rate)

            if FLAGS.opt == 'sgd':
                optimizer = tf.train.MomentumOptimizer(
                    learning_rate=learning_rate,
                    momentum=0.9
                )
            elif FLAGS.opt == 'nadam':
                optimizer = tf.contrib.opt.NadamOptimizer(
                    learning_rate=learning_rate)
            elif FLAGS.opt == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            else:
                raise Exception(f'Undefined Optimizer: {FLAGS.opt}')

            optimizer = tf.contrib.estimator.clip_gradients_by_norm(
                optimizer, 5.0)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=global_steps)
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    return model_fn


def main():

    # flag definitions
    tf.app.flags.DEFINE_string(
        'opt', 'sgd', '''Directory where to log tensorboard summaries.''')
    tf.app.flags.DEFINE_string(
        'loss_type', 'arc', '''Directory where to log tensorboard summaries.''')
    tf.app.flags.DEFINE_boolean('uv', False, '''If include uv channle''')
    tf.app.flags.DEFINE_boolean('coord', False, '''If include uv channle''')
    tf.app.flags.DEFINE_boolean('heatmap', False, '''If include uv channle''')
    from deepmachine.flags import FLAGS

    tf.reset_default_graph()
    NUM_GPUS = len(FLAGS.gpu.split(','))
    BATCH_SIZE = FLAGS.batch_size
    INPUT_SHAPE = 112
    NF = 64
    N_CLASSES = 8631
    LOGDIR = format_folder(FLAGS)
    EPOCH_STEPS = N_CLASSES * 50 // (BATCH_SIZE * NUM_GPUS)

    # configuration
    strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=NUM_GPUS)
    config = tf.estimator.RunConfig(
        train_distribute=strategy,
        save_checkpoints_steps=EPOCH_STEPS,
        save_summary_steps=100,
        keep_checkpoint_max=None,
    )

    # Set up Hooks
    class CustomCallback(tf.train.SessionRunHook):

        def __init__(self, start_step=0, total_epoch=FLAGS.n_epoch, *args, **kwargs):
            self._step = start_step - 1
            self.total_epoch = total_epoch

            return super().__init__(*args, **kwargs)

        def begin(self):
            self.times = []
            self.total_steps = EPOCH_STEPS * self.total_epoch

        def before_run(self, run_context):
            self._step += 1
            self.iter_time_start = time.time()

        def after_run(self, run_context, run_values):
            self.times.append(time.time() - self.iter_time_start)

            if self._step % 20 == 0:
                total_time = sum(self.times)
                avg_time_per_batch = np.mean(self.times[-20:])
                estimate_finishing_time = (
                    self.total_steps - self._step) * avg_time_per_batch
                i_batch = self._step % EPOCH_STEPS
                i_epoch = self._step // EPOCH_STEPS

                print(
                    f"INFO: Epoch [{i_epoch}/{self.total_epoch}], Batch [{i_batch}/{EPOCH_STEPS}]")
                print(
                    f"INFO: Estimate Finishing time: {datetime.timedelta(seconds=estimate_finishing_time)}")
                print(
                    f"INFO: Image/sec: {BATCH_SIZE*NUM_GPUS/avg_time_per_batch}")
                print(f"INFO: Total Time: {datetime.timedelta(seconds=total_time)}")

    # Create the Estimator
    arcface_recognition = tf.estimator.Estimator(
        model_fn=model_optimizer_fn(FLAGS, N_CLASSES, NF, EPOCH_STEPS), model_dir=LOGDIR, config=config)

    # validation data
    writer = tf.summary.FileWriter(LOGDIR)
    validation_file = '/home/dengjiankang/vgg_data/validation/validation.h5py'
    with h5py.File(validation_file) as fd:

        all_labels = fd['label'][:].squeeze()

        pred_input1_fn = tf.estimator.inputs.numpy_input_fn(
            x={
                "image": fd['image_1'][:].astype(np.float32) / 255.,
                "uv": fd['uv_1'][:].astype(np.float32)
            }, shuffle=False
        )

        pred_input2_fn = tf.estimator.inputs.numpy_input_fn(
            x={
                "image": fd['image_2'][:].astype(np.float32) / 255.,
                "uv": fd['uv_2'][:].astype(np.float32)
            }, shuffle=False
        )

    def cos_distance(x,y):
        return x.dot(y)/np.linalg.norm(x)/np.linalg.norm(y)

    # train
    for epoch_i in range(FLAGS.n_epoch):

        arcface_recognition.train(
            get_data_fn(FLAGS, N_CLASSES, INPUT_SHAPE), 
            steps=EPOCH_STEPS, 
            hooks=[CustomCallback(start_step=epoch_i*EPOCH_STEPS)]
        )

        
        pred_results1 = arcface_recognition.predict(input_fn=pred_input1_fn)
        r1 = list(pred_results1)
        r1_emb = np.array([r['embedding'] for r in r1])

        pred_results2 = arcface_recognition.predict(input_fn=pred_input2_fn)
        r2 = list(pred_results2)
        r2_emb = np.array([r['embedding'] for r in r2])

        score = np.array([cos_distance(eb1, eb2) for eb1, eb2 in zip(r1_emb, r2_emb)])
        fpr, tpr, thred = sklearn.metrics.roc_curve(all_labels, score)
        acc = sklearn.metrics.roc_auc_score(all_labels, score)

        print(f"INFO: ACCURACY: {acc}")
        summary = tf.Summary(value=[tf.Summary.Value(tag='accuracy', simple_value=acc)])
        writer.add_summary(summary, epoch_i*EPOCH_STEPS)

        


if __name__ == '__main__':
    main()
