import tensorflow as tf
from deepmachine.flags import FLAGS

slim = tf.contrib.slim


def restore(pretrained_model_checkpoint_path=FLAGS.pretrained_model_checkpoint_path):
    init_fn = None

    if pretrained_model_checkpoint_path:
        print('Loading whole model ...')
        variables_to_restore = slim.get_model_variables()
        init_fn = slim.assign_from_checkpoint_fn(
            pretrained_model_checkpoint_path,
            variables_to_restore,
            ignore_missing_vars=True)
    return init_fn
