import tensorflow as tf

slim = tf.contrib.slim


def restore(pretrained_model_checkpoint_path=None):
    init_fn = None

    if pretrained_model_checkpoint_path:
        print('Loading whole model ...')
        variables_to_restore = slim.get_model_variables()
        init_fn = slim.assign_from_checkpoint_fn(
            pretrained_model_checkpoint_path,
            variables_to_restore,
            ignore_missing_vars=True)
    return init_fn


def restore_gan_generator_from_ckpt(checkpoint_path=None):
    init_fn = None

    if checkpoint_path:
        print('Loading model %s'%checkpoint_path)
        variables_to_restore = slim.get_model_variables()
        variables_to_restore = {var.op.name[10:]: var for var in variables_to_restore if 'generator/' in var.op.name}
        init_fn = slim.assign_from_checkpoint_fn(
            checkpoint_path,
            variables_to_restore,
            ignore_missing_vars=True)
    return init_fn
