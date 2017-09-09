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
