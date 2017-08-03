import numpy as np
import menpo.io as mio
import scipy.io as sio
from io import BytesIO

import tensorflow as tf


def get_jpg_string(im):
    # Gets the serialized jpg from a menpo `Image`.
    fp = BytesIO()
    mio.export_image(im, fp, extension='jpg')
    fp.seek(0)
    return fp.read()


def _int_feauture(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feauture(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feauture(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def image_builder(data):
    image = data['image']

    return {
        'image': _bytes_feauture(get_jpg_string(image)),
        'height': _int_feauture(image.shape[0]),
        'width': _int_feauture(image.shape[1])
    }

def landmark_builder(data):
    visible_pts = data['visible_pts']
    marked_index = data['marked_index']
    image = data['image']

    landmarks = image.landmarks['JOINT'].points
    return {
        'n_landmarks': _int_feauture(landmarks.shape[0]),
        'gt': _bytes_feauture(landmarks.astype(np.float32).tobytes()),
        'visible': _bytes_feauture(np.array(visible_pts).astype(np.int64).tobytes()),
        'marked': _bytes_feauture(np.array(marked_index).astype(np.int64).tobytes()),
    }
