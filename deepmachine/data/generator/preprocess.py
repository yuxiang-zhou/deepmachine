import menpo.io as mio
import numpy as np

def img_preprocessing(img, shape=[256, 256]):

    if img.n_channels == 1:
        img.pixels = np.tile(img.pixels, [3, 1, 1])

    shape = np.array(shape)
    offset = shape // 4

    img = img.resize(shape + offset)

    # random crop
    img = random_crop(img, shape)

    if np.random.rand() > 0.5:
        img = img.mirror()

    return img.pixels_with_channels_at_back() * 2. - 1


def random_crop(img, crop_shape):
    
    offset = np.array(img.shape) - crop_shape
    offset = (np.random.sample([2]) * offset).astype(np.int)
    img = img.crop(offset, offset + crop_shape)
    return img


def img_load(img_path):
    return mio.import_image(img_path)