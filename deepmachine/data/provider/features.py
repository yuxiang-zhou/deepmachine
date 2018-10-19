import tensorflow as tf
from ...utils import union_dict


image_feature = {
    # images
    'image': tf.FixedLenFeature([], tf.string),
    'height': tf.FixedLenFeature([], tf.int64),
    'width': tf.FixedLenFeature([], tf.int64),
}

iuv_feature = {
    # iuv
    'iuv': tf.FixedLenFeature([], tf.string),
    'iuv_height': tf.FixedLenFeature([], tf.int64),
    'iuv_width': tf.FixedLenFeature([], tf.int64),
}
uvxyz_feature = {
    # iuv
    'uvxyz': tf.FixedLenFeature([], tf.string),
    'uvxyz/mask': tf.FixedLenFeature([], tf.string),
    'uvxyz/height': tf.FixedLenFeature([], tf.int64),
    'uvxyz/width': tf.FixedLenFeature([], tf.int64),
}

svs_feature = {
    # svs
    'n_svs': tf.FixedLenFeature([], tf.int64),
    'n_svs_ch': tf.FixedLenFeature([], tf.int64),
    'svs': tf.FixedLenFeature([], tf.string),
}

lms_feature = {
    # landmarks
    'n_landmarks': tf.FixedLenFeature([], tf.int64),
    'gt': tf.FixedLenFeature([], tf.string),
}

FeatureSequence = {
    # sequences
    'frames': tf.VarLenFeature(tf.string),
    'drawings': tf.VarLenFeature(tf.string),

    # meta data
    'height': tf.FixedLenFeature([], tf.int64),
    'width': tf.FixedLenFeature([], tf.int64),
}

FeatureMaskedSequence = {
    # sequences
    'frames': tf.VarLenFeature(tf.string),
    'drawings': tf.VarLenFeature(tf.string),
    'masks': tf.FixedLenFeature([], tf.string),

    # meta data
    'height': tf.FixedLenFeature([], tf.int64),
    'width': tf.FixedLenFeature([], tf.int64),
}

FeatureIUVHMSVS = {
    # customs features
    'visible': tf.FixedLenFeature([], tf.string),
    'marked': tf.FixedLenFeature([], tf.string),
    'scale': tf.FixedLenFeature([], tf.float32),
    ## original infomations
    'original_scale': tf.FixedLenFeature([], tf.float32),
    'original_centre': tf.FixedLenFeature([], tf.string),
    'original_lms': tf.FixedLenFeature([], tf.string),
    ## inverse transform to original landmarks
    'restore_translation': tf.FixedLenFeature([], tf.string),
    'restore_scale': tf.FixedLenFeature([], tf.float32)
}
FeatureIUVHMSVS.update(
    union_dict([image_feature, iuv_feature, svs_feature, lms_feature])
)

FeatureIUVHM = union_dict([image_feature, iuv_feature, lms_feature])
FeatureIUV = union_dict([image_feature, iuv_feature])

FeatureHeatmap = {
    'visible': tf.FixedLenFeature([], tf.string),
    'marked': tf.FixedLenFeature([], tf.string),
}
FeatureHeatmap.update(union_dict([image_feature, lms_feature]))


FeatureRLMS = {
    # landmarks
    'n_landmarks': tf.FixedLenFeature([], tf.int64),
    'rlms': tf.FixedLenFeature([], tf.string),
    'visible': tf.FixedLenFeature([], tf.string),
    'marked': tf.FixedLenFeature([], tf.string),
}
FeatureRLMS.update(union_dict([image_feature]))