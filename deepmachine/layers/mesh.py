import scipy
import numpy as np
import tensorflow as tf
import menpo.io as mio
import keras
import keras.backend as K
import io
from scipy import sparse
from ..utils import get_custom_objects, mesh as graph
from ..layers import Layer, InputSpec


class MeshReLU1B(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Bias and ReLU. One bias per filter."""

        _, _, n_channels = input_shape
        # Create a trainable weight variable for this layer.
        self.bias = self.add_weight(
            name='kernel',
            shape=[1, n_channels],
            initializer='uniform',
            trainable=True)
        super().build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return K.relu(x + self.bias)

    def compute_output_shape(self, input_shape):
        return input_shape


class MeshReLU2B(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Bias and ReLU. One bias per filter."""
        _, n_vertexes, n_channels = input_shape
        # Create a trainable weight variable for this layer.
        self.bias = self.add_weight(
            name='kernel',
            shape=[n_vertexes, n_channels],
            initializer='uniform',
            trainable=True)
        super().build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return K.relu(x + self.bias)

    def compute_output_shape(self, input_shape):
        return input_shape


class MeshPoolTrans(Layer):

    def poolwT(self, x):
        L = self._gl
        Mp = L.shape[0]
        _, M, Fin = x.get_shape().as_list()
        # Rescale transform Matrix L and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)

        x = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x = tf.reshape(x, [M, -1])  # M x Fin*N
        x = tf.sparse_tensor_dense_matmul(L, x)  # Mp x Fin*N
        x = tf.reshape(x, [Mp, Fin, -1])  # Mp x Fin x N
        x = tf.transpose(x, perm=[2, 0, 1])  # N x Mp x Fin

        return x

    def __init__(self, graph_laplacians, **kwargs):
        self._gl = graph_laplacians.astype(np.float32)
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Bias and ReLU. One bias per filter."""
        super().build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return self.poolwT(x)

    def compute_output_shape(self, input_shape):
        Mp = self._gl.shape[0]
        N, _, Fin = input_shape
        return (N, Mp, Fin)

    def get_config(self):
        # serialize sparse matrix
        byte_stream = io.BytesIO()
        sparse.save_npz(byte_stream, self._gl)

        base_config = super().get_config()
        base_config['graph_laplacians'] = byte_stream.getvalue().decode(
            'latin1')
        return base_config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config['graph_laplacians'] = sparse.load_npz(
            io.BytesIO(config['graph_laplacians'].encode('latin1')))
        return cls(**config)


MeshPool = MeshPoolTrans


class MeshConv(Layer):

    def chebyshev5(self, x, L, Fout, nK):
        L = L.astype(np.float32)
        _, M, Fin = x.get_shape().as_list()
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        L = graph.rescale_L(L, lmax=2)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        # Transform to Chebyshev basis
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, -1])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N

        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N
        if nK > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)
            x = concat(x, x1)
        for k in range(2, nK):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = tf.reshape(x, [nK, M, Fin, -1])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # N x M x Fin x K
        x = tf.reshape(x, [-1, Fin*nK])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        W = self._weight_variable
        x = tf.matmul(x, W)  # N*M x Fout
        out = tf.reshape(x, [-1, M, Fout])  # N x M x Fout
        return out

    def __init__(self, graph_laplacians, polynomial_order=6, nf=16, **kwargs):
        self._gl = graph_laplacians
        self._nf = nf
        self._po = polynomial_order
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Bias and ReLU. One bias per filter."""

        _, _, n_channels = input_shape
        # Create a trainable weight variable for this layer.
        self._weight_variable = self.add_weight(
            name='kernel',
            shape=[n_channels * self._po, self._nf],
            initializer='uniform',
            trainable=True)
        super().build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return self.chebyshev5(x, self._gl, self._nf, self._po)

    def compute_output_shape(self, input_shape):
        N, M, _ = input_shape
        return (N, M, self._nf)

    def get_config(self):
        # serialize sparse matrix
        byte_stream = io.BytesIO()
        sparse.save_npz(byte_stream, self._gl)

        base_config = super().get_config()
        base_config['polynomial_order'] = self._po
        base_config['nf'] = self._nf
        base_config['graph_laplacians'] = byte_stream.getvalue().decode(
            'latin1')
        return base_config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config['graph_laplacians'] = sparse.load_npz(
            io.BytesIO(config['graph_laplacians'].encode('latin1')))
        return cls(**config)


get_custom_objects().update({
    'MeshReLU1B': MeshReLU1B,
    'MeshReLU2B': MeshReLU2B,
    'MeshPool': MeshPool,
    'MeshPoolTrans': MeshPoolTrans,
    'MeshConv': MeshConv,
})
