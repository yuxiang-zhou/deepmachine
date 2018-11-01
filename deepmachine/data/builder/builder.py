import numpy as np
import menpo.io as mio
import scipy.io as sio
from io import BytesIO
from menpo.shape import PointCloud
from menpo.image import Image

import tensorflow as tf


def get_jpg_string(im):
    # Gets the serialized jpg from a menpo `Image`.
    if not isinstance(im, Image):
        im = Image.init_from_channels_at_back(im)
    fp = BytesIO()
    mio.export_image(im, fp, extension='jpg')
    fp.seek(0)
    return fp.read()


def int_feauture(value):
    if type(value) is not list:
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feauture(value):
    if type(value) is not list:
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feauture(value):
    if type(value) is not list:
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def image_builder(data, key='image'):
    image = data[key]

    return {
        key: bytes_feauture(get_jpg_string(image)),
        '%s/height'%key: int_feauture(image.shape[0]),
        '%s/width'%key: int_feauture(image.shape[1])
    }

def uvxyz_builder(data):
    image = data['uvxyz']
    mask = data['uvxyz/mask']
    return {
        'uvxyz': bytes_feauture(image.pixels_with_channels_at_back().astype(np.float32).tobytes()),
        'uvxyz/mask': bytes_feauture(get_jpg_string(mask)),
        'uvxyz/height': int_feauture(image.shape[0]),
        'uvxyz/width': int_feauture(image.shape[1])
    }

def relative_landmark_builder(data):
    visible_pts = data['visible_pts']
    marked_index = data['marked_index']
    landmarks = data['rlms']
    return {
        'n_landmarks': int_feauture(landmarks.shape[0]),
        'rlms': bytes_feauture(landmarks.astype(np.float32).tobytes()),
        'visible': bytes_feauture(np.array(visible_pts).astype(np.int64).tobytes()),
        'marked': bytes_feauture(np.array(marked_index).astype(np.int64).tobytes()),
    }


def landmark_builder(data, key='landmarks', visible_label=None, marked_label=None):
    
    landmarks = data[key]
    if isinstance(landmarks, PointCloud):
        landmarks = landmarks.points
        
    results = {
        '%s/count'%key: int_feauture(landmarks.shape[0]),
        key: bytes_feauture(landmarks.astype(np.float32).tobytes()),
    }

    if visible_label:
        visible_pts = data[visible_label]
        results.update({
            '%s/visible'%key: bytes_feauture(np.array(visible_pts).astype(np.int64).tobytes()),
        })

    if marked_label:
        marked_index = data[marked_label]
        results.update({
            '%s/marked'%key: bytes_feauture(np.array(marked_index).astype(np.int64).tobytes()),
        })

    return results

def tensor_builder(data, key='data', dtype=np.float32):
    m = data[key]

    return {
        '%s'%key: bytes_feauture(m.astype(dtype).tobytes()),
        '%s/height'%key: int_feauture(m.shape[0]),
        '%s/width'%key: int_feauture(m.shape[1]),
        '%s/depth'%key: int_feauture(m.shape[2])
    }

def matrix_builder(data, key='data', dtype=np.float32):
    m = data[key]

    return {
        '%s'%key: bytes_feauture(m.astype(dtype).tobytes()),
        '%s/height'%key: int_feauture(m.shape[0]),
        '%s/width'%key: int_feauture(m.shape[1])
    }

def array_builder(data, key='data', dtype=np.float32):
    m = data[key]

    return {
        '%s'%key: bytes_feauture(m.astype(dtype).tobytes()),
        '%s/size'%key: int_feauture(m.shape[0]),
    }

def iuv_builder(data):
    image = data['iuv']

    return {
        'iuv': bytes_feauture(image.pixels_with_channels_at_back().astype(np.float32).tobytes()),
        'iuv_height': int_feauture(image.shape[0]),
        'iuv_width': int_feauture(image.shape[1])
    }
