import keras
import tensorflow as tf
import numpy as np
import menpo.io as mio
import menpo
from scipy.interpolate import interp1d
import scipy as sp
from keras import backend as K
from matplotlib import pyplot as plt
from pathlib import Path
from scipy.io import loadmat
from menpo.image import Image
from menpo.shape import PointCloud
from menpo.transform import Translation, Scale

from .channel_transform import sample_colours_from_colourmap

ResizeMethod = tf.image.ResizeMethod

# tf functions


def tf_caffe_preprocess(image):
    VGG_MEAN = np.array([102.9801, 115.9465, 122.7717])
    # RGB -> BGR
    image = tf.reverse(image, [False, False, True])
    # Subtract VGG training mean across all channels
    image = image - VGG_MEAN.reshape([1, 1, 3])
    return image


def tf_rotate_points(points, image, angle):

    s = tf.shape(image)
    image_center = tf.to_float(s[:2]) / 2.

    # center coordinates since rotation center is supposed to be in the image center
    points_centered = points - image_center

    rot_matrix = tf.dynamic_stitch([[0], [1], [2], [3]], [tf.cos(
        angle), -tf.sin(angle), tf.sin(angle), tf.cos(angle)])
    rot_matrix = tf.reshape(rot_matrix, shape=[2, 2])

    points_centered_rot = tf.matmul(rot_matrix, tf.transpose(points_centered))

    return tf.transpose(points_centered_rot) + image_center


def tf_lms_to_heatmap(lms, h, w, n_landmarks, marked_index, sigma=5):
    xs, ys = tf.meshgrid(tf.range(0., tf.to_float(w)),
                         tf.range(0., tf.to_float(h)))
    gaussian = (1. / (sigma * np.sqrt(2. * np.pi)))

    def gaussian_fn(lms):
        y, x, idx = tf.unstack(lms)
        idx = tf.to_int32(idx)

        def run_true():
            return tf.exp(-0.5 * (tf.pow(ys - y, 2) + tf.pow(xs - x, 2)) *
                          tf.pow(1. / sigma, 2.)) * gaussian * 17.

        def run_false():
            return tf.zeros((h, w))

        return tf.cond(tf.reduce_any(tf.equal(marked_index, idx)), run_true, run_false)

    img_hm = tf.stack(tf.map_fn(gaussian_fn, tf.concat(
        [lms, tf.to_float(tf.range(0, n_landmarks))[..., None]], 1)))

    return img_hm


def tf_heatmap_to_lms(heatmap):
    hs = tf.argmax(tf.reduce_max(heatmap, 2), 1)
    ws = tf.argmax(tf.reduce_max(heatmap, 1), 1)
    lms = tf.transpose(tf.to_float(tf.stack([hs, ws])), perm=[1, 2, 0])

    return lms


def tf_image_batch_to_grid(images, col_size=4):
    image_shape = tf.shape(images)

    batch_size = image_shape[0]
    image_height = image_shape[1]
    image_width = image_shape[2]
    image_channels = image_shape[3]

    w = col_size
    h = batch_size // w

    tfimg = images[:w * h]
    tfimg = tf.reshape(
        tfimg, [w, h * image_height, image_width, image_channels])
    tfimg = tf.reshape(
        tf.transpose(tfimg, [1, 0, 2, 3]), [h * image_height, w * image_width, image_channels])

    return tfimg


def tf_image_patch_around_lms(image, lms, patch_size=32, dtype=tf.float32):

    pad_size = patch_size // 2 + 1
    lms = tf.to_int32(lms) + tf.constant([pad_size, pad_size])
    image = tf.pad(image, [[pad_size, pad_size], [pad_size, pad_size], [0, 0]])

    def crop(x):
        return tf.image.crop_to_bounding_box(image, x[0] - pad_size, x[1] - pad_size, patch_size, patch_size)

    image = tf.concat(tf.unstack(tf.map_fn(crop, lms, dtype=dtype)), axis=-1)

    return image


def tf_records_iterator(path, feature=None):

    record_iterator = tf.python_io.tf_record_iterator(path=path)
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        yield example.features.feature


def tf_logits_to_heatmap(logits, num_classes):
    """Generates a coloured heatmap from the keypoint logits.

    Args:
        features: A `Tensor` of dimensions [num_batch, height, width, FLAGS.n_landmarks + 1].
    """

    keypoint_colours = np.array(
        [
            plt.cm.spectral(x)
            for x in np.linspace(0, 1, num_classes + 1)
        ])[..., :3].astype(np.float32)

    prediction = tf.nn.softmax(logits)
    heatmap = tf.matmul(tf.reshape(
        prediction, (-1, num_classes + 1)), keypoint_colours)
    heatmap = tf.reshape(heatmap, (tf.shape(prediction)[0],
                                   tf.shape(prediction)[1],
                                   tf.shape(prediction)[2], 3))
    return heatmap


def tf_logits_to_landmarks(keypoints):
    is_background = tf.equal(keypoints, 0)
    ones = tf.to_float(tf.ones_like(is_background))
    zeros = tf.to_float(tf.zeros_like(is_background))

    return tf.where(is_background, zeros, ones) * 255


def tf_keypts_encoding(keypoints, num_classes):
    keypoints = tf.to_int32(keypoints)
    keypoints = tf.reshape(keypoints, (-1,))
    keypoints = tf.layers.one_hot_encoding(
        keypoints, num_classes=num_classes + 1)
    return keypoints


def tf_get_weight(keypoints, mask=None, ng_w=0.01, ps_w=1.0):
    is_background = tf.equal(keypoints, 0)
    ones = tf.to_float(tf.ones_like(is_background))
    weights = tf.where(is_background, ones * ng_w, ones * ps_w)
    # if mask is not None:
    #     weights *= tf.to_float(mask)

    return weights


def tf_atan2(y, x):
    angle = tf.where(tf.greater(x, 0.0), tf.atan(y / x), tf.zeros_like(x))
    angle = tf.where(tf.greater(y, 0.0), 0.5 * np.pi - tf.atan(x / y), angle)
    angle = tf.where(tf.less(y, 0.0), -0.5 * np.pi - tf.atan(x / y), angle)
    angle = tf.where(tf.less(x, 0.0), tf.atan(y / x) + np.pi, angle)
    angle = tf.where(tf.logical_and(tf.equal(x, 0.0), tf.equal(y, 0.0)),
                     np.nan * tf.zeros_like(x), angle)

    indices = tf.where(tf.less(angle, 0.0))
    updated_values = tf.gather_nd(angle, indices) + (2 * np.pi)
    update = tf.SparseTensor(indices, updated_values, angle.get_shape())
    update_dense = tf.sparse_tensor_to_dense(update)

    return angle + update_dense


def tf_n_channel_rgb(inputs, n_feature, colour_set='jet'):
    cm = sample_colours_from_colourmap(
        n_feature, colour_set).astype(np.float32)
    tf_cm = tf.constant(cm)
    tf_img = tf.tensordot(inputs, tf_cm, axes=1)

    return tf_img


def tf_iuv_rgb(tf_iuv, n_feature=26, colour_set='jet'):

    tf_iuv_class = tf_iuv[..., :n_feature]
    tf_iuv_class = tf.argmax(tf_iuv_class, axis=-1)
    tf_iuv_class = tf.one_hot(tf_iuv_class, n_feature)

    tf_u = tf_iuv_class * tf_iuv[..., n_feature:n_feature*2]
    tf_v = tf_iuv_class * tf_iuv[..., n_feature*2:]

    tf_u = tf_n_channel_rgb(tf_u, n_feature)
    tf_v = tf_n_channel_rgb(tf_v, n_feature)

    tf_img = (tf_u + tf_v) / 2. / 255.

    return tf_img


def tf_ced_accuracy(t, dists):
    # Head	 Shoulder	Elbow	Wrist	Hip	   Knee	   Ankle
    pts_r = tf.transpose(
        tf.gather(tf.transpose(dists), [8, 12, 11, 10, 2, 1, 0]))
    pts_l = tf.transpose(
        tf.gather(tf.transpose(dists), [9, 13, 14, 15, 3, 4, 5]))
    part_pckh = (tf.to_int32(pts_r <= t) + tf.to_int32(pts_l <= t)) / 2

    return tf.concat([part_pckh, tf.reduce_sum(tf.to_int32(dists <= t), 1)[..., None] / tf.shape(dists)[1]], 1)


def tf_normalized_point_to_point_error(preds, gts, factor=1):
    dists = tf.sqrt(tf.reduce_sum(tf.pow(preds - gts, 2),
                                  reduction_indices=-1)) / factor
    return dists

def tf_pckh(preds, gts, scales):
    t_range = np.arange(0, 0.51, 0.01)
    dists = tf_normalized_point_to_point_error(preds, gts, factor=scales)
    return tf_ced_accuracy(0.5, dists)

## Edge Detection --------------
def tf_canny(img_tensor, minRate=0.10, maxRate=0.40, remove_high_val=False, return_raw_edges=False):
    """ STEP-0 (Preprocessing): 
        1. Scale the tensor values to the expected range ([0,1]) 
        2. If 'preserve_size': As TensorFlow will pad by 0s for padding='SAME',
                               it is better to pad by the same values of the borders.
                               (This is to avoid considering the borders as edges)
    """

    GAUS_KERNEL = img_tensor.get_shape().as_list()[-1]
    GAUS_SIGMA  = 1.2
    def Gaussian_Filter(kernel_size=GAUS_KERNEL, sigma=GAUS_SIGMA): #Default: Filter_shape = [5,5]
    # --> Reference: https://en.wikipedia.org/wiki/Canny_edge_detector#Gaussian_filter
        k = (kernel_size-1)//2 
        filters = []
        sigma_2 = sigma**2
        for i in range(kernel_size):
            filter_row = []
            for j in range(kernel_size):
                Hij = np.exp(-((i+1-(k+1))**2 + (j+1-(k+1))**2)/(2*sigma_2))/(2*np.pi*sigma_2)
                filter_row.append(Hij)
            filters.append(filter_row)
        
        return np.asarray(filters).reshape(kernel_size,kernel_size,1,1).transpose([2,0,1,3])

    """
    NOTE: 	All variables are initialized first for reducing proccessing time.
    """
    gaussian_filter = tf.constant(Gaussian_Filter(), tf.float32) #STEP-1
    h_filter = tf.reshape(tf.constant([[-1,0,1],[-2,0,2],[-1,0,1]], tf.float32), [3,3,1,1])	#STEP-2
    v_filter = tf.reshape(tf.constant([[1,2,1],[0,0,0],[-1,-2,-1]], tf.float32), [3,3,1,1])	#STEP-2

    np_filter_0 = np.zeros((3,3,1,2))
    np_filter_0[1,0,0,0], np_filter_0[1,2,0,1] = 1,1 ### Left & Right
    # print(np_filter_0)
    filter_0 = tf.constant(np_filter_0, tf.float32)
    np_filter_90 = np.zeros((3,3,1,2))
    np_filter_90[0,1,0,0], np_filter_90[2,1,0,1] = 1,1 ### Top & Bottom
    filter_90 = tf.constant(np_filter_90, tf.float32)
    np_filter_45 = np.zeros((3,3,1,2))
    np_filter_45[0,2,0,0], np_filter_45[2,0,0,1] = 1,1 ### Top-Right & Bottom-Left
    filter_45 = tf.constant(np_filter_45, tf.float32)
    np_filter_135 = np.zeros((3,3,1,2))
    np_filter_135[0,0,0,0], np_filter_135[2,2,0,1] = 1,1 ### Top-Left & Bottom-Right
    filter_135 = tf.constant(np_filter_135, tf.float32)
        
    np_filter_sure = np.ones([3,3,1,1]); np_filter_sure[1,1,0,0] = 0
    filter_sure = tf.constant(np_filter_sure, tf.float32)
    border_paddings = tf.constant([[0,0],[1,1],[1,1],[0,0]])

    def FourAngles(d):
        d0   = tf.to_float(tf.greater_equal(d,157.5))+tf.to_float(tf.less(d,22.5))
        d45  = tf.to_float(tf.greater_equal(d,22.5))*tf.to_float(tf.less(d,67.5))
        d90  = tf.to_float(tf.greater_equal(d,67.5))*tf.to_float(tf.less(d,112.5))
        d135 = tf.to_float(tf.greater_equal(d,112.5))*tf.to_float(tf.less(d,157.5))
        # return {'d0':d0, 'd45':d45, 'd90':d90, 'd135':d135}
        return (d0,d45,d90,d135)

    """ STEP-1: Noise reduction with Gaussian filter """
    x_gaussian = tf.nn.convolution(img_tensor, gaussian_filter, padding='SAME')
    ### Below is a heuristic to remove the intensity gradient inside a cloud ###
#     if remove_high_val: x_gaussian = tf.clip_by_value(x_gaussian, 0, MAX/2)
    
    
    """ STEP-2: Calculation of Horizontal and Vertical derivatives  with Sobel operator 
        --> Reference: https://en.wikipedia.org/wiki/Sobel_operator	
    """
    Gx = tf.nn.convolution(x_gaussian, h_filter, padding='SAME')
    Gy = tf.nn.convolution(x_gaussian, v_filter, padding='SAME')
    G = tf.sqrt(tf.square(Gx) + tf.square(Gy))
    BIG_PHI = tf.atan2(Gy,Gx)
    BIG_PHI = (BIG_PHI*180/np.pi)%180 ### Convert from Radian to Degree
    D_0,D_45,D_90,D_135 = FourAngles(BIG_PHI)### Round the directions to 0, 45, 90, 135 (only take the masks)
    
    
    """ STEP-3: NON-Maximum Suppression
        --> Reference: https://stackoverflow.com/questions/46553662/conditional-value-on-tensor-relative-to-element-neighbors
    """
    
    """ 3.1-Selecting Edge-Pixels on the Horizontal direction """
    targetPixels_0 = tf.nn.convolution(G, filter_0, padding='SAME')
    isGreater_0 = tf.to_float(tf.greater(G*D_0, targetPixels_0))
    isMax_0 = isGreater_0[:,:,:,0:1]*isGreater_0[:,:,:,1:2]
    ### Note: Need to keep 4 dimensions (index [:,:,:,0] is 3 dimensions) ###
    
    """ 3.2-Selecting Edge-Pixels on the Vertical direction """
    targetPixels_90 = tf.nn.convolution(G, filter_90, padding='SAME')
    isGreater_90 = tf.to_float(tf.greater(G*D_90, targetPixels_90))
    isMax_90 = isGreater_90[:,:,:,0:1]*isGreater_90[:,:,:,1:2]
    
    """ 3.3-Selecting Edge-Pixels on the Diag-45 direction """
    targetPixels_45 = tf.nn.convolution(G, filter_45, padding='SAME')
    isGreater_45 = tf.to_float(tf.greater(G*D_45, targetPixels_45))
    isMax_45 = isGreater_45[:,:,:,0:1]*isGreater_45[:,:,:,1:2]
    
    """ 3.4-Selecting Edge-Pixels on the Diag-135 direction """
    targetPixels_135 = tf.nn.convolution(G, filter_135, padding='SAME')
    isGreater_135 = tf.to_float(tf.greater(G*D_135, targetPixels_135))
    isMax_135 = isGreater_135[:,:,:,0:1]*isGreater_135[:,:,:,1:2]
    
    """ 3.5-Merging Edges on Horizontal-Vertical and Diagonal directions """
    edges_raw = G*(isMax_0 + isMax_90 + isMax_45 + isMax_135)
    edges_raw = tf.clip_by_value(edges_raw, 0, 1)
    
    return edges_raw
