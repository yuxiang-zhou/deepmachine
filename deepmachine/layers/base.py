# import tensorflow as tf

# class Conv2D(tf.keras.layers.Conv2D):
#     def __init__(self, *args, activation=tf.keras.layers.LeakyReLU, batch_normalize=tf.keras.layers.BatchNormalization, **kwargs):
#         self.activation = activation
#         self.batchnorm = batch_normalize
#         super().__init__(*args, **kwargs)

#     def call(self, inputs):
#         outputs = super().call(inputs)
#         if self.batchnorm is not None:
#             outputs = self.batchnorm()(outputs)
#         if self.activation is not None:
#             outputs = self.activation()(outputs)
        
#         return outputs