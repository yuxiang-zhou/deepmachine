from pathlib import Path
import tensorflow as tf


class TFRecordBuilder(object):

    def __init__(self):
        self._feature_builder = []

    def add_feature_builder(self, builder):
        self._feature_builder.append(builder)

    def generate(self, iterator, store_path):
        # load annotations
        writer = tf.python_io.TFRecordWriter(store_path)

        for data in iterator:
            # build feature
            feature = {}
            try:
                for op in self._feature_builder:
                    f = op(data)
                    for k in f:
                        feature[k] = f[k]

                # construct the Example proto boject
                example = tf.train.Example(
                    features=tf.train.Features(feature=feature)
                )

                # use the proto object to serialize the example to a string
                serialized = example.SerializeToString()
    
                # write the serialized object to disk
                writer.write(serialized)
            except Exception as e:
                print(e)

        writer.close()
