import tensorflow as tf

slim = tf.contrib.slim
from .. import utils


def pose_pckh(data_eps, network_eps):
    summary_ops = []

    # get data
    gt_heatmap = data_eps['heatmap']
    pred_heatmap, _ = network_eps

    summary_ops.append(tf.summary.image(
        'images', data_eps['inputs'], max_outputs=4))
    summary_ops.append(tf.summary.image('gt/heatmap', tf.reduce_sum(gt_heatmap, -1)
                                        [..., None] * 255.0, max_outputs=4))

    # get landmarks
    gt_lms = utils.tf_heatmap_to_lms(gt_heatmap)
    pred_lms = utils.tf_heatmap_to_lms(pred_heatmap)

    scales = tf.norm(gt_lms[:, 9] - gt_lms[:, 8])

    # calculate accuracy
    accuracy_all = utils.pckh(pred_lms, gt_lms, scales)

    # streaming_mean
    metrics_to_values, metrics_to_updates = slim.metrics.aggregate_metric_map({

        "accuracy/pckh_All": slim.metrics.streaming_mean(accuracy_all[:, -1]),
        "accuracy/pckh_Head": slim.metrics.streaming_mean(accuracy_all[:, 0]),
        "accuracy/pckh_Shoulder": slim.metrics.streaming_mean(accuracy_all[:, 1]),
        "accuracy/pckh_Elbow": slim.metrics.streaming_mean(accuracy_all[:, 2]),
        "accuracy/pckh_Wrist": slim.metrics.streaming_mean(accuracy_all[:, 3]),
        "accuracy/pckh_Hip": slim.metrics.streaming_mean(accuracy_all[:, 4]),
        "accuracy/pckh_Knee": slim.metrics.streaming_mean(accuracy_all[:, 5]),
        "accuracy/pckh_Ankle": slim.metrics.streaming_mean(accuracy_all[:, 6])
    })

    # Define the streaming summaries to write
    for metric_name, metric_value in metrics_to_values.items():
        op = tf.summary.scalar(metric_name, metric_value)
        op = tf.Print(op, [metric_value], metric_name)
        summary_ops.append(op)

    summary_ops.append(tf.summary.scalar(
        'accuracy/running_pckh', tf.reduce_mean(accuracy_all[:, -1])))
    summary_ops.append(tf.summary.image(
        'predictions/landmark-regression',
        tf.reduce_sum(pred_heatmap, -1)[..., None] * 255.0,
        max_outputs=4))

    return list(metrics_to_updates.values()), summary_ops


def face_nmse(data_eps, network_eps):

    summary_ops = []

    # get data
    gt_heatmap = data_eps['heatmap']
    pred_heatmap, _ = network_eps

    summary_ops.append(tf.summary.image(
        'images', data_eps['inputs'], max_outputs=4))
    summary_ops.append(tf.summary.image('gt/heatmap', tf.reduce_sum(gt_heatmap, -1)
                                        [..., None] * 255.0, max_outputs=4))

    # get landmarks
    gt_lms = utils.tf_heatmap_to_lms(gt_heatmap)
    pred_lms = utils.tf_heatmap_to_lms(pred_heatmap)

    scales = tf.norm(tf.reduce_mean(
        gt_lms[:, 36:42], 0) - tf.reduce_mean(gt_lms[:, 42:48], 0))

    # calculate accuracy
    accuracy = utils.normalized_point_to_point_error(pred_lms, gt_lms, scales)

    # streaming_mean
    metrics_to_values, metrics_to_updates = slim.metrics.aggregate_metric_map({
        "accuracy/normalized_mean_square_error": slim.metrics.streaming_mean(accuracy),
    })

    # Define the streaming summaries to write
    for metric_name, metric_value in metrics_to_values.items():
        op = tf.summary.scalar(metric_name, metric_value)
        op = tf.Print(op, [metric_value], metric_name)
        summary_ops.append(op)

    summary_ops.append(tf.summary.scalar(
        'accuracy/running_nmse', tf.reduce_mean(accuracy)))
    summary_ops.append(tf.summary.image(
        'predictions/landmark-regression',
        tf.reduce_sum(pred_heatmap, -1)[..., None] * 255.0,
        max_outputs=4))

    return list(metrics_to_updates.values()), summary_ops
