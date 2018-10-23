import numpy as np
import menpo.io as mio
import math
import keras
from scipy.interpolate import interp1d
import scipy as sp
import binascii
from menpo.compatibility import unicode
from struct import pack as struct_pack

from pathlib import Path
from scipy.io import loadmat
from menpo.image import Image
from menpo.shape import PointCloud, TriMesh, ColouredTriMesh
from menpo.transform import Translation, Scale, AlignmentSimilarity
from menpo.model import pca

def one_hot(a, n_parts):
    a = a.astype(np.int32)
    b = np.zeros((len(a), n_parts))
    b[np.arange(len(a)), a] = 1
    return b

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

    index = np.argmax(iuv[..., :n_channel], axis=-
                      1).squeeze().astype(np.ushort)

    u = iuv[..., n_channel:n_channel * 2]
    v = iuv[..., n_channel * 2:]

    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)

    for i in range(n_channel):
        u[index != i, i] = 0
        v[index != i, i] = 0

    return np.array([index, u.max(axis=-1), v.max(axis=-1)])


def iuv_normalise(iuv):
    i = iuv[..., :2]
    u = iuv[..., 2:4]
    v = iuv[..., 4:]

    i = np.stack([
        i.argmin(axis=-1),
        i.argmax(axis=-1)
    ], axis=-1)
    u[:,:,0] *= 0
    u[:,:,1] *= i[:,:,1]
    v[:,:,0] *= 0
    v[:,:,1] *= i[:,:,1]
    u = u.clip(0,1)
    v = v.clip(0,1)

    iuv_new = np.concatenate([i,u,v], axis=-1)
    
    return iuv_new


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


def channels_to_grid(pixels, n_col=4):

    h, w, n_channel = pixels.shape
    n_col = np.min([n_col, n_channel])
    n_row = n_channel // n_col
    grid = pixels[:, :, :n_col *
                  n_row].reshape([h, w, n_col, n_row]).transpose([2, 3, 0, 1])
    grid = np.concatenate(grid, axis=2)
    grid = np.concatenate(grid, axis=0)

    return grid


def rgb_iuv(rgb):
    # formation
    iuv_mask = rgb[..., 0]
    n_parts = int(np.max(iuv_mask) + 1)
    iuv_one_hot = one_hot(iuv_mask.flatten(), n_parts).reshape(
        iuv_mask.shape + (n_parts,))

    # normalised uv
    uv = rgb[..., 1:] / 255. if np.max(rgb[..., 1:]) > 1 else rgb[..., 1:]
    u = iuv_one_hot * uv[..., 0][..., None]
    v = iuv_one_hot * uv[..., 1][..., None]

    iuv = np.concatenate([iuv_one_hot, u, v], 2)

    return iuv


svs_rgb = channels_to_rgb


def lms_to_heatmap(lms, h, w, sigma=5):
    xs, ys = np.meshgrid(np.arange(0., w),
                         np.arange(0., h))
    gaussian = (1. / (sigma * np.sqrt(2. * np.pi)))

    def gaussian_fn(l):
        y, x = l

        return np.exp(-0.5 * (np.power(ys - y, 2) + np.power(xs - x, 2)) *
                      np.power(1. / sigma, 2.)) * gaussian * 17.

    img_hm = np.stack(list(map(
        gaussian_fn,
        lms
    )))

    return img_hm


def heatmap_to_lms(heatmap):
    hs = np.argmax(np.max(heatmap, 1), 0)
    ws = np.argmax(np.max(heatmap, 0), 0)
    lms = np.stack([hs, ws]).T

    return lms