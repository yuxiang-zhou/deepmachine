import numpy as np
import scipy
import menpo.io as mio
import functools
import time
import traceback
import tensorflow as tf

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


slim = tf.contrib.slim

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


def get_ae_inception_hourglass_pose():
    # create machine
    model = deepmachine.DeepMachine(
        network_op=functools.partial(
            networks.base.StackedHourglassAE,
            n_channels=16,
            n_stacks=2,
            deconv='transpose+conv',
            bottleneck='bottleneck_inception'
        )
    )

    # add losses
    def loss_stacked_landmark_regression(data_eps, network_eps, alpha=1.0, heatmap_weight=500):
        gt_heatmap = data_eps['heatmap']
        _, states = network_eps

        weight_hm = utils.get_weight(gt_heatmap, ng_w=0.1, ps_w=1) * heatmap_weight
        l2norm = 0

        for idx_stack, predictions in enumerate(states[:2]):

            # landmark-regression losses
            l2norm += slim.losses.mean_squared_error(
                predictions, gt_heatmap, weights=weight_hm * alpha)

            # losses summaries
            tf.summary.scalar('losses/lms_stack_%02d' % idx_stack, l2norm)

        tf.losses.add_loss(l2norm, loss_collection='regression_loss')

    model.add_loss_op(functools.partial(loss_stacked_landmark_regression, alpha=1.0))
    model.add_loss_op(functools.partial(losses.loss_landmark_regression, alpha=1.0))
    model.add_loss_op(functools.partial(losses.loss_landmark_reconstruction, alpha=1.0))

    # add summaries
    def summary_HG(data_eps, network_eps, is_training=True, n_channel=16):
        predictions, status = network_eps

        tf.summary.image(
            'predictions/HG_landmarks',
            utils.tf_n_channel_rgb(status[1], n_channel),
            max_outputs=3)


    model.add_summary_op(summary.summary_landmarks)
    model.add_summary_op(summary_HG)


    # set evaluation op
    model.eval_op = ops.eval.pose_pckh
    model.train_op = ops.train.adam_ae

    return model


def get_pose_auto_encoder():
    # create machine
    model = deepmachine.DeepMachine(
        network_op=functools.partial(
            networks.base.AutoEncoder,
            n_channels=16,
            deconv='transpose+conv'
        )
    )

    # add losses
    model.add_loss_op(functools.partial(losses.loss_landmark_reconstruction, alpha=1.0))

    # add summaries
    model.add_summary_op(summary.summary_landmarks)

    # set evaluation op
    model.eval_op = ops.eval.pose_pckh
    model.train_op = ops.train.adam

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
            deconv='transpose+bn',
            bottleneck='bottleneck_inception',
        )
    )

    # add losses
    model.add_loss_op(functools.partial(
        losses.loss_landmark_regression, collection='generator_loss', alpha=5
    ))
    model.add_loss_op(functools.partial(
        losses.loss_posegan_generator, alpha=1
    ))
    model.add_loss_op(functools.partial(
        losses.loss_posegan_discriminator, alpha=1
    ))

    # add summaries
    model.add_summary_op(summary.summary_landmarks)

    # set ops
    model.train_op = ops.train.gan
    model.init_op = ops.init.restore_gan_generator_from_ckpt
    model.eval_op = ops.eval.pose_pckh

    return model