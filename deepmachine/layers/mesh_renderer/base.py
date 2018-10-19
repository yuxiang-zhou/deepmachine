import scipy
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K

from ...utils import get_custom_objects
from ...layers import Lambda
from . import mesh_renderer

def Renderer(image_height=256, image_width=256, **kwargs):

    kwargs['image_height'] = image_height
    kwargs['image_width'] = image_width

    return Lambda(mesh_renderer.mesh_renderer, output_shape=[image_height, image_width, 4], mask=None, arguments=kwargs)

get_custom_objects().update({
    'Renderer': Renderer,
    'mesh_renderer': mesh_renderer
})