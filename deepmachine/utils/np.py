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


def union_dict(dicts):
    udict = {}
    for d in dicts:
        udict.update(d)
    return udict


def fill_hole_row(r):
    p_s = p_e = 0
    while p_e < len(r) and p_s < len(r):
        if r[p_s] == 0 and r[p_e] > 0 or p_e + 1 == len(r):
            start = r[p_s-1] if p_s > 0 else 0
            end = r[p_e] if r[p_e] > 0 else start
            r[p_s:p_e] = np.linspace(start, end, p_e-p_s)
            p_s = p_e = p_e + 1
        else:
            if r[p_e] == 0:
                p_e += 1

            if r[p_s] > 0:
                p_s += 1
                p_e = p_s


def fill_hole(img):
    img_hf = img.copy()
    for r in img_hf:
        fill_hole_row(r)

    for r in img_hf.T:
        fill_hole_row(r)

    return img_hf


def rotation_matrix(angle, direction, point=None):
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = np.array(direction).astype(np.float64)
    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array([[0.0,         -direction[2],  direction[1]],
                   [direction[2], 0.0,          -direction[0]],
                   [-direction[1], direction[0],  0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


def uv_template_from_mesh(mesh, uv_shape=(256, 256)):
    from menpo3d.unwrap import optimal_cylindrical_unwrap

    UV_template = optimal_cylindrical_unwrap(mesh).apply(mesh)
    UV_template.points = UV_template.points[:, [1, 0]]
    UV_template.points[:, 0] = UV_template.points[:,
                                                  0].max() - UV_template.points[:, 0]

    UV_template.points -= UV_template.points.min(axis=0)
    UV_template.points /= UV_template.points.max(axis=0)
    UV_template.points *= np.array([uv_shape])

    return UV_template


def uv_in_img(image, mesh, uv_template):
    from menpo3d.rasterize import rasterize_mesh

    shape = image.shape
    u_mesh = ColouredTriMesh(
        mesh.points,
        trilist=mesh.trilist,
        colours=uv_template.points[:, 0, None])
    u_image = rasterize_mesh(u_mesh, shape)
    v_mesh = ColouredTriMesh(
        mesh.points,
        trilist=mesh.trilist,
        colours=uv_template.points[:, 1, None])
    v_image = rasterize_mesh(v_mesh, shape)

    IUV_image = Image(np.concatenate(
        [u_image.mask.pixels, u_image.pixels / 255., v_image.pixels / 255.]).clip(0, 1))

    return IUV_image


def project_landmarks_to_shape_model(landmarks):
    final = []

    for lms in landmarks:
        lms = PointCloud(lms)
        similarity = AlignmentSimilarity(pca.global_transform.source, lms)
        projected_target = similarity.pseudoinverse().apply(lms)
        target = pca.model.reconstruct(projected_target)
        target = similarity.apply(target)
        final.append(target.points)

    return np.array(final).astype(np.float32)


def caffe_preprocess(image):
    VGG_MEAN = np.array([102.9801, 115.9465, 122.7717])

    # RGB -> BGR
    pixels = image.pixels[[2, 1, 0]]
    # Subtract VGG training mean across all channels
    pixels = pixels - VGG_MEAN.reshape([3, 1, 1])
    pixels = pixels.astype(np.float32, copy=False)
    return pixels


def rescale_image(image, stride_width=64):
    # make sure smallest size is 600 pixels wide & dimensions are (k * stride_width) + 1
    height, width = image.shape

    # Taken from 'szross'
    scale_up = 625. / min(height, width)
    scale_cap = 961. / max(height, width)
    scale_up = min(scale_up, scale_cap)
    new_height = stride_width * round((height * scale_up) / stride_width) + 1
    new_width = stride_width * round((width * scale_up) / stride_width) + 1
    image, tr = image.resize([new_height, new_width], return_transform=True)
    image.inverse_tr = tr
    return image


def frankotchellappa(dzdx, dzdy):
    from numpy.fft import ifftshift, fft2, ifft2
    rows, cols = dzdx.shape
    # The following sets up matrices specifying frequencies in the x and y
    # directions corresponding to the Fourier transforms of the gradient
    # data.  They range from -0.5 cycles/pixel to + 0.5 cycles/pixel.
    # The scaling of this is irrelevant as long as it represents a full
    # circle domain. This is functionally equivalent to any constant * pi
    pi_over_2 = np.pi / 2.0
    row_grid = np.linspace(-pi_over_2, pi_over_2, rows)
    col_grid = np.linspace(-pi_over_2, pi_over_2, cols)
    wy, wx = np.meshgrid(row_grid, col_grid, indexing='ij')

    # Quadrant shift to put zero frequency at the appropriate edge
    wx = ifftshift(wx)
    wy = ifftshift(wy)

    # Fourier transforms of gradients
    DZDX = fft2(dzdx)
    DZDY = fft2(dzdy)

    # Integrate in the frequency domain by phase shifting by pi/2 and
    # weighting the Fourier coefficients by their frequencies in x and y and
    # then dividing by the squared frequency
    denom = (wx ** 2 + wy ** 2)
    Z = (-1j * wx * DZDX - 1j * wy * DZDY) / denom
    Z = np.nan_to_num(Z)
    return np.real(ifft2(Z))


def line(image, x0, y0, x1, y1, color):
    steep = False
    if x0 < 0 or x0 >= 400 or x1 < 0 or x1 >= 400 or y0 < 0 or y0 >= 400 or y1 < 0 or y1 >= 400:
        return

    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        steep = True

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    for x in range(int(x0), int(x1) + 1):
        t = (x - x0) / float(x1 - x0)
        y = y0 * (1 - t) + y1 * t
        if steep:
            image[x, int(y)] = color
        else:
            image[int(y), x] = color


def draw_landmarks(img, lms):
    try:
        img = img.copy()

        for i, part in enumerate(parts_68[1:]):
            circular = []

            if i in (4, 5, 6, 7):
                circular = [part[0]]

            for p1, p2 in zip(part, list(part[1:]) + circular):
                p1, p2 = lms[p1], lms[p2]

                line(img, p2[1], p2[0], p1[1], p1[0], 1)
    except:
        pass
    return img


def batch_draw_landmarks(imgs, lms):
    return np.array([draw_landmarks(img, l) for img, l in zip(imgs, lms)])


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


def crop_image_bounding_box(img, bbox, res, base=200., order=1):

    center = bbox.centre()
    bmin, bmax = bbox.bounds()
    scale = np.linalg.norm(bmax - bmin) / base

    return crop_image(img, center, scale, res, order=order)


def crop_image(img, center, scale, res, base=384., order=1):
    h = base * scale

    t = Translation(
        [
            res[0] * (-center[0] / h + .5),
            res[1] * (-center[1] / h + .5)
        ]).compose_after(Scale((res[0] / h, res[1] / h))).pseudoinverse()

    # Upper left point
    ul = np.floor(t.apply([0, 0]))
    # Bottom right point
    br = np.ceil(t.apply(res).astype(np.int))

    # crop and rescale

    cimg, trans = img.warp_to_shape(
        br - ul, Translation(-(br - ul) / 2 + (br + ul) / 2), return_transform=True)
    c_scale = np.min(cimg.shape) / np.mean(res)
    new_img = cimg.rescale(1 / c_scale, order=order).resize(res, order=order)

    trans = trans.compose_after(Scale([c_scale, c_scale]))

    return new_img, trans, c_scale


def arclen_polyl(cnt):

    tang = np.diff(cnt, axis=0)
    seg_len = np.sqrt(np.power(tang[:, 0], 2) + np.power(tang[:, 1], 2))
    seg_len = np.hstack((0, seg_len))
    alparam = np.cumsum(seg_len)
    cntLen = alparam[-1]
    return alparam, cntLen


def interpolate(points, step, kind='slinear'):
    alparam, cntLen = arclen_polyl(points)

    f_x = interp1d(
        alparam, points[:, 0], kind=kind
    )

    f_y = interp1d(
        alparam, points[:, 1], kind=kind
    )

    points_dense_x = f_x(np.arange(0, cntLen, step))
    points_dense_y = f_y(np.arange(0, cntLen, step))

    points_dense = np.hstack((
        points_dense_x[:, None], points_dense_y[:, None]
    ))

    return points_dense


def multi_channel_svs(svs_pts, h, w, groups, c=3):
    msvs = Image.init_blank((h, w), n_channels=len(groups))
    for ch, g in enumerate(groups):
        if len(g):
            msvs.pixels[ch, ...] = svs_shape(
                svs_pts, h, w, groups=[g], c=c).pixels[0]
    msvs.pixels /= np.max(msvs.pixels)
    return msvs


def svs_shape(pc, xr, yr, groups=None, c=1):
    store_image = Image.init_blank((xr, yr))
    ni = binary_shape(pc, xr, yr, groups)
    store_image.pixels[0, :, :] = sp.ndimage.filters.gaussian_filter(
        np.squeeze(ni.pixels), c)
    return store_image


def binary_shape(pc, xr, yr, groups=None):
    return sample_points(pc.points, xr, yr, groups)


def sample_points(target, range_x, range_y, edge=None, x=0, y=0):
    ret_img = Image.init_blank((range_x, range_y))

    if edge is None:
        for pts in target:
            ret_img.pixels[0, pts[0]-y, pts[1]-x] = 1.
    else:
        for eg in edge:
            for pts in interpolate(target[eg], 0.1):
                try:
                    ret_img.pixels[0, int(pts[0]-y), int(pts[1]-x)] = 1.
                except:
                    pass
                    # print('Index out of Bound')

    return ret_img


def max_epoch(path):
    path = Path(path).glob('*weights.*.hdf5')
    return np.max([-1]+[int(p.suffixes[0][1:]) for p in path])


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

    colours = sample_colours_from_colourmap(
        n_channel, colour_set
    )

    return (u.dot(colours) / 255. + v.dot(colours) / 255.) / 2.


def one_hot(a, n_parts):
    a = a.astype(np.int32)
    b = np.zeros((len(a), n_parts))
    b[np.arange(len(a)), a] = 1
    return b


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


svs_rgb = channels_to_rgb

def enqueue_generator(data_generator, use_multiprocessing=True, workers=4, max_queue_size=256):
    enqueuer = keras.utils.data_utils.GeneratorEnqueuer(
        data_generator,
        use_multiprocessing=use_multiprocessing,
        wait_time=0)
    enqueuer.start(workers=workers, max_queue_size=max_queue_size)

    output_generator = enqueuer.get()

    return output_generator