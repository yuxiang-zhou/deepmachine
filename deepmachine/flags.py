import os
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
# catching extra 'f' flag from jupyter notebook
tf.app.flags.DEFINE_string('f', '', 'kernel')

# training parameters
tf.app.flags.DEFINE_string('gpu', '',
                    '''gpu id to use''')

tf.app.flags.DEFINE_string('dataset_path', None,
                    '''Directory where to load datas and checkpoint.''')

tf.app.flags.DEFINE_string('logdir', None,
                    '''Directory where to log tensorboard summaries.''')

tf.app.flags.DEFINE_integer('batch_size', 32,
                    '''Batch Size''')

tf.app.flags.DEFINE_integer('verbose', 2,
                    '''Stdout infomation details''')

tf.app.flags.DEFINE_integer('no_thread', 4,
                    '''Number of data reading threads''')
                    
tf.app.flags.DEFINE_float('lr', 2e-4,
                    '''Learning rate''')
tf.app.flags.DEFINE_float('lr_decay', 0.99,
                    '''Learning rate decay = learning_rate * lr_decay ** epoch''')

tf.app.flags.mark_flags_as_required([
    'dataset_path', 'logdir'
])

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu