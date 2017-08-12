import tensorflow as tf
import numpy as np

from deepmachine.flags import FLAGS
import deepmachine.models.stackedHG as models

slim = tf.contrib.slim


def TFGraph(
    inputs,
    graph_def_path,
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
            tf.import_graph_def(od_graph_def, name='')
