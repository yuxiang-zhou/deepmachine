import numpy as np
import scipy
import menpo.io as mio
import functools
import time
import traceback

from menpo.image import Image
from menpo.shape import PointCloud
# deep modules
import deepmachine

from .. import utils
from .. import losses
from .. import summary
from .. import data_provider
from .. import ops
from .. import networks
from ..flags import FLAGS


def get_dense_pose_net():
    # create machine
    model = deepmachine.DeepMachine(
        network_op=functools.partial(
            networks.pose.DensePoseTorch, deconv='transpose+conv')
    )

    # add losses
    model.add_loss_op(losses.loss_landmark_regression)
    model.add_loss_op(losses.loss_iuv_regression)

    # add summaries
    model.add_summary_op(summary.summary_landmarks)
    model.add_summary_op(summary.summary_iuv)

    # set evaluation op
    model.eval_op = ops.eval.pose_pckh

    return model


def get_dense_pose_net_tf():
    # create machine
    model = deepmachine.DeepMachine(
        network_op=functools.partial(
            networks.pose.DensePoseTF, deconv='transpose+conv')
    )

    # add losses
    model.add_loss_op(losses.loss_landmark_regression)
    model.add_loss_op(losses.loss_iuv_regression)

    # add summaries
    model.add_summary_op(summary.summary_landmarks)
    model.add_summary_op(summary.summary_iuv)

    # set evaluation op
    model.eval_op = ops.eval.pose_pckh

    return model


def get_hourglass_pose():
    # create machine
    model = deepmachine.DeepMachine(
        network_op=functools.partial(
            networks.base.StackedHourglass,
            n_channels=16,
            n_stacks=2,
            deconv='transpose+conv'
        )
    )

    # add losses
    model.add_loss_op(losses.loss_stacked_landmark_regression)

    # add summaries
    model.add_summary_op(summary.summary_landmarks)

    # set evaluation op
    model.eval_op = ops.eval.pose_pckh

    return model


def get_inception_hourglass_pose():
    # create machine
    model = deepmachine.DeepMachine(
        network_op=functools.partial(
            networks.base.StackedHourglass,
            n_channels=16,
            n_stacks=2,
            deconv='transpose+conv',
            bottleneck='bottleneck_inception'
        )
    )

    # add losses
    model.add_loss_op(losses.loss_stacked_landmark_regression)

    # add summaries
    model.add_summary_op(summary.summary_landmarks)

    # set evaluation op
    model.eval_op = ops.eval.pose_pckh

    return model


def get_se_inception_hourglass_pose():
    # create machine
    model = deepmachine.DeepMachine(
        network_op=functools.partial(
            networks.base.StackedHourglass,
            n_channels=16,
            n_stacks=2,
            deconv='transpose+conv',
            bottleneck='bottleneck_inception_SE'
        )
    )

    # add losses
    model.add_loss_op(losses.loss_stacked_landmark_regression)

    # add summaries
    model.add_summary_op(summary.summary_landmarks)

    # set evaluation op
    model.eval_op = ops.eval.pose_pckh

    return model


def get_gan_pose():
    # create machine
    model = deepmachine.DeepMachine(
        network_op=functools.partial(
            networks.gan.GAN,
            n_channels=16,
            n_stacks=2,
            deconv='transpose+conv',
            key_discriminant='heatmap'
        )
    )

    # add losses
    model.add_loss_op(losses.loss_generator)
    model.add_loss_op(losses.loss_discriminator)

    # add summaries
    model.add_summary_op(summary.summary_landmarks)

    # set ops
    model.train_op = ops.train.gan
    model.init_op = ops.init.restore_gan_generator_from_ckpt

    return model


# faces

def get_densereg_face(n_classes=11, use_regression=False):
    # create machine
    model = deepmachine.DeepMachine(
        network_op=functools.partial(
            networks.face.DenseRegFace,
            n_classes=FLAGS.quantization_step + 1,
            deconv='transpose+conv'
        )
    )

    # add losses
    model.add_loss_op(losses.loss_uv_classification)

    # add summaries
    model.add_summary_op(summary.summary_uv)

    return model


def get_hourglass_face():
    # create machine
    model = deepmachine.DeepMachine(
        network_op=functools.partial(
            networks.base.StackedHourglass,
            n_channels=68,
            n_stacks=2,
            deconv='transpose+conv'
        )
    )

    # add losses
    model.add_loss_op(losses.loss_stacked_landmark_regression)

    # add summaries
    model.add_summary_op(summary.summary_landmarks)

    # set evaluation op
    model.eval_op = ops.eval.pose_pckh

    return model


def get_dense_cascade_face():
    # create machine
    model = deepmachine.DeepMachine(
        network_op=functools.partial(
            networks.face.DenseFaceCascade,
            n_features=(FLAGS.quantization_step + 1) * 2,
            deconv='transpose+conv'
        )
    )

    # add losses
    model.add_loss_op(losses.loss_landmark_regression)
    model.add_loss_op(losses.loss_uv_classification)

    # add summaries
    model.add_summary_op(summary.summary_landmarks)

    # set evaluation op
    model.eval_op = ops.eval.pose_pckh

    return model


# general

def get_cyclegan():
    # create machine
    model = deepmachine.DeepMachine(
        network_op=functools.partial(
            networks.gan.CycleGAN,
            n_channels=3
        )
    )

    # add losses
    model.add_loss_op(losses.loss_cyclegan_discriminator)
    model.add_loss_op(losses.loss_cyclegan_generator)

    # add summaries
    model.add_summary_op(summary.summary_cyclegan)

    # set ops
    model.train_op = ops.train.cyclegan

    return model