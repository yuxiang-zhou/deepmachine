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
from deepmachine.networks import pose, face
from .. import utils
from .. import losses
from .. import summary
from .. import data_provider
from .. import ops

def get_dense_pose_net():
    # create machine
    model = deepmachine.DeepMachine(
        network_op=functools.partial(pose.DensePoseTorch, deconv='transpose+conv')
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
        network_op=functools.partial(pose.DensePoseTorch, deconv='transpose+conv')
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
            pose.StackedHourglass,
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


def get_densereg_face(n_classes=11, use_regression=False):
    # create machine
    model = deepmachine.DeepMachine(
        network_op=functools.partial(
            face.DenseRegFace,
            n_classes=FLAGS.quantization_step+1,
            deconv='transpose+conv'
        )
    )

    # add losses
    model.add_loss_op(losses.loss_landmark_regression)
    model.add_loss_op(losses.loss_uv_classification)

    # add summaries
    model.add_summary_op(summary.summary_landmarks)
    model.add_summary_op(summary.summary_uv)

    # set evaluation op
    # model.eval_op = ops.eval.pose_pckh

    return model
