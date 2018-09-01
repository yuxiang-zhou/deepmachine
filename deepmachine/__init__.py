import os
os.environ["KERAS_BACKEND"] = 'tensorflow'

import keras
from keras import Model, initializers, optimizers, activations
from keras.models import model_from_config, model_from_json, model_from_yaml, save_model

import tensorflow as tf
from .base import *
from . import utils
from . import networks
from . import layers
from . import callbacks
from . import losses

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
