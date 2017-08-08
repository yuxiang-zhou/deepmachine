import tensorflow as tf

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS
# data parameters
tf.app.flags.DEFINE_string('dataset_dir', '',
                           '''Directory where to load datas '''
                           '''and checkpoint.''')

# training parameters
tf.app.flags.DEFINE_string('train_model', '',
                            '''training mode''')
tf.app.flags.DEFINE_string('train_dir', 'ckpt/train',
                           '''Directory where to write event logs '''
                           '''and checkpoint.''')
tf.app.flags.DEFINE_integer('batch_size', 1, '''The batch size to use.''')
tf.app.flags.DEFINE_float('initial_learning_rate', 0.0005,
                          '''Initial learning rate.''')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.96,
                          '''Learning rate decay factor.''')
tf.app.flags.DEFINE_float('learning_rate_decay_step', 15000,
                          '''Learning rate decay factor.''')
tf.app.flags.DEFINE_integer('number_of_steps', None,
                            '''The max number of gradient steps to take during training. If the value is left as None, training proceeds indefinitely.''')
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            '''How many preprocess threads to use.''')
tf.app.flags.DEFINE_integer('moving_average_ckpt', 0,
                            '''moving_average_ckpt''')
tf.app.flags.DEFINE_integer('log_every_n_steps', 1,
                            '''log_every_n_steps''')
tf.app.flags.DEFINE_integer('logging_level', 20,
                            '''logging_level''')
tf.app.flags.DEFINE_string('data_keys', 'inputs,heatmap,iuv',
                            '''type of data to load''')
tf.app.flags.DEFINE_string('db_provider', 'DensePoseProvider',
                            '''db_provider''')




tf.app.flags.DEFINE_string('eval_dir', '',
                           '''Directory where to write event logs '''
                           '''and checkpoint.''')
tf.app.flags.DEFINE_integer('eval_size', 200, '''The batch size to use.''')

tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                           '''If specified, restore this pretrained model '''
                           '''before beginning any training.''')


# DenseReg parameters
tf.app.flags.DEFINE_integer('quantization_step', 10,
                            '''How many quantization_step to use.''')
