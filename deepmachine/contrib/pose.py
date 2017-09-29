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
            networks.pose.DensePoseTF,
            deconv='transpose+conv+relu',
            bottleneck='bottleneck_inception'
        )
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


def get_dense_pose_net_old():
    # create machine
    model = deepmachine.DeepMachine(
        network_op=functools.partial(
            networks.pose.DensePoseTF,
            deconv='transpose+conv',
            bottleneck='bottleneck'
        )
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
            networks.gan.PoseGAN,
            n_channels=16,
            deconv='transpose+conv',
            bottleneck='bottleneck_inception',
        )
    )

    # add losses
    model.add_loss_op(functools.partial(
        losses.loss_landmark_regression, collection='generator_loss'
    ))
    model.add_loss_op(functools.partial(
        losses.loss_generator, alpha=10.0
    ))
    model.add_loss_op(functools.partial(
        losses.loss_discriminator, alpha=10.0
    ))

    # add summaries
    model.add_summary_op(summary.summary_landmarks)

    # set ops
    model.train_op = ops.train.gan
    model.init_op = ops.init.restore_gan_generator_from_ckpt
    model.eval_op = ops.eval.pose_pckh

    return model