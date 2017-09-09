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


def get_cyclegan_hg():
    # create machine
    model = deepmachine.DeepMachine(
        network_op=functools.partial(
            networks.gan.CycleGANHG,
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
