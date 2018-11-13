import tensorflow as tf
import keras
import keras.backend as K
from keras.utils import get_custom_objects
from keras.layers import Layer, initializers, regularizers, constraints


class ArcDense(Layer):

    def __init__(
        self,
        units,
        kernel_initializer='glorot_uniform',
        kernel_regularizer=None,
        kernel_constraint=None,
        **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super().__init__(**kwargs)
        self.units = units
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

    def call(self, inputs):
        # l2 normalize parameters
        norm_x = K.l2_normalize(inputs)
        norm_w = K.l2_normalize(self.kernel)

        # compute arc distance
        output = K.dot(norm_x, norm_w)
        return output

    def compute_output_shape(self, input_shape):
        return (None, self.units)

    def get_config(self):
        config = {
            'units': self.units,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


get_custom_objects().update({
    'ArcDense': ArcDense
})