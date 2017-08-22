import tensorflow as tf
import numpy as np

from deepmachine.flags import FLAGS
import deepmachine.models.stackedHG as models

slim = tf.contrib.slim


def TFGraph(
    inputs,
    graph_def_path,
    input_tensor='inputs',
    output_tensors=[],
    is_training=True,
    **kwargs
):
    batch_size = tf.shape(inputs)[0]
    height = tf.shape(inputs)[1]
    width = tf.shape(inputs)[2]
    channels = tf.shape(inputs)[3]

    states = {}

    with slim.arg_scope([slim.batch_norm, slim.layers.dropout], is_training=is_training):

        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_def_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(
                od_graph_def,
                input_map={"%s:0" % input_tensor: inputs},
                name=''
            )

    current_graph = tf.get_default_graph()
    output = [current_graph.get_tensor_by_name(
        '%s:0' % tensor_name) for tensor_name in output_tensors]

    return output
