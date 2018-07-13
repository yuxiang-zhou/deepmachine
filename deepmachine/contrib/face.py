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



# faces

def get_hourglass_face():
    # create machine
    model = deepmachine.DeepMachine(
        network_op=functools.partial(
            networks.base.StackedHourglass,
            n_channels=68,
            n_stacks=2,
            deconv='transpose+conv+relu'
        )
    )

    # add losses
    model.add_loss_op(losses.loss_stacked_landmark_regression)

    # add summaries
    model.add_summary_op(summary.summary_landmarks)

    # set evaluation op
    model.eval_op = ops.eval.face_nmse

    return model


def get_dense_cascade_face():
    # create machine
    model = deepmachine.DeepMachine(
        network_op=functools.partial(
            networks.face.DenseFaceCascade,
            n_features=(FLAGS.quantization_step + 1) * 2,
            deconv='transpose+conv+relu'
        )
    )

    # add losses
    model.add_loss_op(losses.loss_landmark_regression)
    model.add_loss_op(losses.loss_uv_classification)

    # add summaries
    model.add_summary_op(summary.summary_landmarks)

    # set evaluation op
    model.eval_op = ops.eval.face_nmse

    return model

def get_face_iuv_auto_encoder():
    # create machine
    model = deepmachine.DeepMachine(
        network_op=functools.partial(
            networks.base.AutoEncoder,
            n_channels=6,
            n_features=128,
            deconv='transpose',
            encoder_key='iuv'
        )
    )

    # add losses
    model.add_loss_op(functools.partial(losses.loss_iuv_regression, alpha=1.0, n_feature=2))

    # add summaries
    model.add_summary_op(functools.partial(summary.summary_iuv, n_feature=2))

    # set train_op
    model.train_op = ops.train.adam

    return model