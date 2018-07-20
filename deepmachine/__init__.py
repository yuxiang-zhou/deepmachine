import tensorflow as tf
from .base import *
from . import utils
from . import networks
from . import layers
from . import callbacks
from . import losses
initializers = tf.keras.initializers
optimizers = tf.keras.optimizers

Model = tf.keras.Model

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
