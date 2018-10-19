import tensorflow as tf
import keras
import keras.backend as K
from keras.utils import get_custom_objects
from . import helper
from .. import utils


def loss_heatmap_regression(y_true, y_pred, heatmap_weight=500):

    # landmark-regression losses
    weight_hm = utils.tf_get_weight(
        y_true, ng_w=0.1, ps_w=1.0) * heatmap_weight
    l2norm = tf.losses.mean_squared_error(
        y_pred, y_true, weights=weight_hm)

    tf.summary.scalar('losses/heatmap', l2norm)

    return l2norm


def loss_iuv_regression(y_true, y_pred):
    n_feature = y_pred.shape.as_list()[-1] // 3

    # mask index cross-entropy loss
    uv_pred_idx = y_pred[..., :n_feature]  # 26 channels
    uv_gt_idx = y_true[..., :n_feature]  # 26 channels

    celoss = tf.losses.softmax_cross_entropy(logits = uv_pred_idx, onehot_labels = uv_gt_idx)

    # uv regression losses
    uv_pred = y_pred[..., n_feature:]  # 52 channels
    uv_gt = y_true[..., n_feature:]  # 52 channels

    weight_hm = uv_gt_idx * 100
    l1smooth_U = helper.smooth_l1(
        uv_pred[..., :n_feature], uv_gt[..., :n_feature], weights=weight_hm)
    l1smooth_V = helper.smooth_l1(
        uv_pred[..., n_feature:], uv_gt[..., n_feature:], weights=weight_hm)

    # losses summaries
    tf.summary.scalar('losses/u_y_pred', l1smooth_U)
    tf.summary.scalar('losses/v_y_pred', l1smooth_V)
    tf.summary.scalar('losses/uv_cross_entropy', celoss)

    return celoss + l1smooth_U + l1smooth_V


def loss_kl(z_mean, z_log_var):
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    return kl_loss


get_custom_objects().update({
    'loss_heatmap_regression': loss_heatmap_regression,
    'loss_iuv_regression': loss_iuv_regression,
    'loss_kl': loss_kl
})
