import numpy as np
import binascii
import tensorflow as tf
from menpo.compatibility import unicode
from struct import pack as struct_pack


def rgb2hex(rgb):
    return '#' + binascii.hexlify(struct_pack('BBB', *rgb)).decode('ascii')


def decode_colour(colour):
    if not (isinstance(colour, str) or isinstance(colour, unicode)):
        # we assume that RGB was passed in. Convert it to unicode hex
        return rgb2hex(colour)
    else:
        return str(colour)


def sample_colours_from_colourmap(n_colours, colour_map):
    import matplotlib.pyplot as plt
    cm = plt.get_cmap(colour_map)
    colours = []
    for i in range(n_colours):
        c = cm(1. * i / n_colours)[:3]
        colours.append(decode_colour([int(i * 255) for i in c]))

    return np.array([hex_to_rgb(x) for x in colours])


def iuv_rgb(iuv, colour_set='jet'):
    iuv = iuv.squeeze()
    n_channel = iuv.shape[-1] // 3

    index = np.argmax(iuv[..., :n_channel], axis=-1).squeeze().astype(np.ushort)

    u = iuv[..., n_channel:n_channel * 2]
    v = iuv[..., n_channel * 2:]

    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)

    for i in range(n_channel):
        u[index != i, i] = 0
        v[index != i, i] = 0

    colours = sample_colours_from_colourmap(
        n_channel, colour_set
    )

    return (u.dot(colours) / 255. + v.dot(colours) / 255.) / 2.


def one_hot(a):
    a = a.astype(np.int32)
    n_parts = np.max(a) + 1
    b = np.zeros((len(a), n_parts))
    b[np.arange(len(a)), a] = 1
    return b


def rgb_iuv(rgb):
    # formation
    iuv_mask = rgb[..., 0]
    n_parts = int(np.max(iuv_mask) + 1)
    iuv_one_hot = one_hot(iuv_mask.flatten()).reshape(iuv_mask.shape + (n_parts,))

    # normalised uv
    uv = rgb[..., 1:] / 255. if np.max(rgb[..., 1:]) > 1 else rgb[..., 1:]
    u = iuv_one_hot * uv[..., 0][..., None]
    v = iuv_one_hot * uv[..., 1][..., None]

    iuv = np.concatenate([iuv_one_hot, u, v], 2)

    return iuv


def hex_to_rgb(hex_str):
    hex_str = hex_str.strip()

    if hex_str[0] == '#':
        hex_str = hex_str[1:]

    if len(hex_str) != 6:
        raise ValueError('Input #{} is not in #RRGGBB format.'.format(hex_str))

    r, g, b = hex_str[:2], hex_str[2:4], hex_str[4:]
    rgb = [int(n, base=16) for n in (r, g, b)]
    return np.array(rgb)


def channels_to_rgb(pixels,
            colour_set='jet'):
    colours = sample_colours_from_colourmap(
        pixels.shape[-1], colour_set
    )

    return pixels.dot(colours) / 255.

svs_rgb = channels_to_rgb


def tf_n_channel_rgb(inputs, n_feature, colour_set='jet'):
    cm = sample_colours_from_colourmap(n_feature, colour_set).astype(np.float32)
    tf_cm = tf.constant(cm)
    tf_img = tf.tensordot(inputs, tf_cm, axes=1)

    return tf_img


def tf_iuv_rgb(tf_iuv, n_feature=26, colour_set='jet'):
   
    tf_iuv_class = tf_iuv[...,:n_feature]
    tf_iuv_class = tf.argmax(tf_iuv_class, axis=-1)
    tf_iuv_class = tf.one_hot(tf_iuv_class, n_feature)
    
    tf_u = tf_iuv_class * tf_iuv[...,n_feature:n_feature*2]
    tf_v = tf_iuv_class * tf_iuv[...,n_feature*2:]
    
    tf_u = tf_n_channel_rgb(tf_u, n_feature)
    tf_v = tf_n_channel_rgb(tf_v, n_feature)
    
    tf_img = (tf_u + tf_v) / 2. / 255.

    return tf_img


