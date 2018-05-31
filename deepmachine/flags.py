import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# catching extra 'f' flag from jupyter notebook
tf.app.flags.DEFINE_string('f', '', 'kernel')

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
tf.app.flags.DEFINE_float('initial_learning_rate', 0.0001,
                          '''Initial learning rate.''')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.97,
                          '''Learning rate decay factor.''')
tf.app.flags.DEFINE_integer('number_of_epochs', 100,
                            '''The max number of gradient steps to take during training. If the value is left as None, training proceeds indefinitely.''')
tf.app.flags.DEFINE_integer('moving_average_ckpt', 0,
                            '''moving_average_ckpt''')
tf.app.flags.DEFINE_integer('log_every_n_steps', 1,
                            '''log_every_n_steps''')
tf.app.flags.DEFINE_integer('logging_level', 20,
                            '''logging_level''')
tf.app.flags.DEFINE_integer('db_size', 0,
                            '''db_size''')

# evaluation parameters
tf.app.flags.DEFINE_string('eval_dir', '',
                           '''Directory where to write event logs '''
                           '''and checkpoint.''')
tf.app.flags.DEFINE_integer('eval_size', 200, '''The batch size to use.''')

tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                           '''If specified, restore this pretrained model '''
                           '''before beginning any training.''')

# data parameters
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            '''How many preprocess threads to use.''')                            
tf.app.flags.DEFINE_string('data_keys', 'inputs,heatmap,iuv',
                           '''type of data to load''')
tf.app.flags.DEFINE_string('db_provider', 'DensePoseProvider',
                           '''db_provider''')


# DenseReg parameters
tf.app.flags.DEFINE_integer('quantization_step', 10,
                            '''How many quantization_step to use.''')

tf.app.flags.DEFINE_integer('n_landmarks', 68,
                            '''How many quantization_step to use.''')
