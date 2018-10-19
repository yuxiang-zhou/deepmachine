import tensorflow as tf
import numpy as np
from functools import partial
from keras.models import Model
from .. import layers
from ..base import K
from .module import conv2d, deconv2d, conv2dt

def MeshEncoder(inputs, embeding, graph_laplacians, downsampling_matrices, polynomial_order=6, filter_list=[16,16,16,32], **kwargs):
    
    net = inputs
    for nf, nl, nd in zip(filter_list, graph_laplacians, downsampling_matrices):
        
        net = layers.MeshConv(nl, nf=nf, polynomial_order=polynomial_order, **kwargs)(net)
        net = layers.MeshReLU1B()(net)
        net = layers.MeshPool(nd)(net)

    # Fully connected hidden layers.
    net = layers.Flatten()(net)
    net = layers.Dense(embeding)(net)
    return net

def MeshDecoder(inputs, out_channel, graph_laplacians, adj_matrices, upsamling_matrices, polynomial_order=6, filter_list=[16,16,16,16], **kwargs):
    pool_size = list(map(lambda x:x.shape[0], adj_matrices))
    net = inputs
    net = layers.Dense(pool_size[-1] * filter_list[-1])(net)
    net = layers.Reshape([pool_size[-1], filter_list[-1]])(net)

    for nf, nl, nu in zip(filter_list[::-1], graph_laplacians[-2::-1], upsamling_matrices[::-1]):
        net = layers.MeshPoolTrans(nu)(net)
        net = layers.MeshConv(nl, nf=nf, polynomial_order=polynomial_order, **kwargs)(net)
        net = layers.MeshReLU1B()(net)

    net = layers.MeshConv(graph_laplacians[0], nf=out_channel, polynomial_order=polynomial_order, **kwargs)(net)

    return net