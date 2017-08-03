import numpy as np
import binascii
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
    return colours


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

    colours = np.array(
        [hex_to_rgb(x)
         for x
         in sample_colours_from_colourmap(
            n_channel, colour_set
        )])

    return (u.dot(colours) / 255. + v.dot(colours) / 255.) / 2.


def hex_to_rgb(hex_str):
    hex_str = hex_str.strip()

    if hex_str[0] == '#':
        hex_str = hex_str[1:]

    if len(hex_str) != 6:
        raise ValueError('Input #{} is not in #RRGGBB format.'.format(hex_str))

    r, g, b = hex_str[:2], hex_str[2:4], hex_str[4:]
    rgb = [int(n, base=16) for n in (r, g, b)]
    return np.array(rgb)


def svs_rgb(pixels,
            pts=None,
            fid=None,
            crop=False,
            render_pts=True,
            alpha=1):
    colours = np.array([hex_to_rgb(x)
                        for x in sample_colours_from_colourmap(7, 'Set2')])

    return pixels.dot(colours) / 255.
