import tensorflow as tf
from .base import DeepMachine, DeepMachineK
from . import utils
from . import networks
from . import layers
initializers = tf.keras.initializers
optimizers = tf.keras.optimizers

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
