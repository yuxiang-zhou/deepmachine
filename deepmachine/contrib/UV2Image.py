
# coding: utf-8

# ### DeepMachine Version

import tensorflow as tf

tf.app.flags.DEFINE_float('gen_loss_weight', 1, 'l1 loss to the reconstruction')
tf.app.flags.DEFINE_float('disc_loss_weight', 1, 'l1 loss to the reconstruction')
tf.app.flags.DEFINE_float('l1_loss_weight', 100, 'l1 loss to the reconstruction')

from functools import partial
import numpy as np
import functools
import deepmachine as dm
from deepmachine import data_provider as dp
from deepmachine import losses
from deepmachine import networks
from deepmachine.flags import FLAGS
from deepmachine.models import stackedHG

import pickle
from menpo.visualize import print_progress
from pathlib import Path

slim = tf.contrib.slim




EPS = 1e-12
batch_size = 8
hps = {
    'gen_loss_weight': FLAGS.gen_loss_weight,
    'disc_loss_weight': FLAGS.disc_loss_weight,
    'l1_loss_weight': FLAGS.l1_loss_weight,
    'learning_rate': FLAGS.initial_learning_rate,
}
img_data_path = '/vol/atlas/homes/jiankang_share/Stylianos/CASIA/CASIA_image/'
uv_data_path = '/vol/atlas/homes/jiankang_share/Stylianos/CASIA/CASIA_fittings_3dmm/'




class TFUV2ImageProvider(dp.Provider):
    def __init__(self,
                uv_dirpath,
                image_dirpath,
                batch_size=1,
                augmentation=False,
                no_processes=16,
                resolvers={}):

        self._uv_dirpath = uv_dirpath
        self._image_dirpath = image_dirpath
        
        self._batch_size = batch_size
        self._augmentation = augmentation
        self.no_processes = no_processes
        self._key_resolver = resolvers

        tmp_list_path = Path('/homes/yz4009/tmp/UV2ImageProvider.pkl')
        
        if not tmp_list_path.exists():
            id_filelist = []
            uv_filelist = []
            image_filelist = []
            mask_filelist = []
            db_path = Path(self._uv_dirpath)
            for gt_uv_path in print_progress(list(db_path.glob('*/*_uv.jpg'))):
                image_id = gt_uv_path.stem.replace('_uv','')
                key = gt_uv_path.parent.stem

                gt_img_path = Path(self._image_dirpath)/key/(image_id+'.jpg')
                gt_mask_path = Path(self._uv_dirpath)/key/(image_id+'_mask.jpg')
                
                uv_filelist.append(str(gt_uv_path))
                image_filelist.append(str(gt_img_path))
                id_filelist.append('{}_{}'.format(key, image_id))
                mask_filelist.append(str(gt_mask_path))

            self._data = [id_filelist, uv_filelist, image_filelist, mask_filelist]
            with open(tmp_list_path.as_posix(),'wb') as f:
                pickle.dump(self._data, f)
        else:
            with open(tmp_list_path.as_posix(),'rb') as f:
                self._data = pickle.load(f)

        self._count = len(self._data[0])
        

    def size(self):
        return self._count

    def get(self, *keys):
        [id_filelist, uv_filelist, image_filelist, mask_filelist] = self._data

        tf_uv_filelist = tf.convert_to_tensor(uv_filelist)
        tf_image_filelist = tf.convert_to_tensor(image_filelist)
        tf_id_filelist = tf.convert_to_tensor(id_filelist)
        tf_mask_filelist = tf.convert_to_tensor(mask_filelist)
        
        n_items = len(uv_filelist)
        print('Number of items: %d' % n_items)

        producer = tf.train.range_input_producer(n_items, capacity=n_items)

        queue = None
        enqueue_ops = []
        for _ in range(self.no_processes):
            tf_filepath_id = producer.dequeue()
            tf_uv_path = tf_uv_filelist[tf_filepath_id]
            tf_image_path = tf_image_filelist[tf_filepath_id]
            tf_mask_path = tf_mask_filelist[tf_filepath_id]
            tf_file_contents = {
                'image': tf.read_file(tf_image_path),
                'mask': tf.read_file(tf_mask_path),
                'uv': tf.read_file(tf_uv_path),
                'id': tf_id_filelist[tf_filepath_id]
            }

            augmentation_args = self._random_augmentation()

            tensors = []
            for key in keys:
                fn = functools.partial(
                   self._key_resolver[key],
                   aug=self._augmentation,
                   aug_args=augmentation_args
                )
                tensors.append(fn(tf_file_contents))

            shapes = [x.get_shape() for x in tensors]

            if queue is None:
                dtypes = [x.dtype for x in tensors]
                queue = tf.FIFOQueue(
                    capacity=200,
                    dtypes=dtypes, name='fifoqueue')

        enqueue_ops.append(queue.enqueue(tensors))

        qr = tf.train.QueueRunner(queue, enqueue_ops)
        tf.train.add_queue_runner(qr)

        tensors_dequeue = queue.dequeue()

        for t, s in zip(tensors_dequeue, shapes):
            t.set_shape(s)

        inputs_batch = tf.train.batch(
            tensors_dequeue, batch_size=self._batch_size, enqueue_many=False)

        return {key:inputs_batch[key_id] for key_id, key in enumerate(keys)}
        




def image_resolver(f, *argv, **kwargs):
    # get image
    tf_image = tf.image.decode_jpeg(f['image'], channels=3)
        
    tf_image =  tf.image.resize_images(
        tf.image.convert_image_dtype(tf_image, tf.float32), [256,256])
        
    tf_image *= 2
    tf_image -= 1
    
    tf_image.set_shape([256,256,3])
    
    return tf_image


def uv_resolver(f, *argv, **kwargs):
    # get image
    tf_image = tf.image.decode_jpeg(f['uv'], channels=3)
        
    tf_image =  tf.image.resize_images(
        tf.image.convert_image_dtype(tf_image, tf.float32), [256,256])
    
    tf_image *= 2
    tf_image -= 1
    
    tf_image.set_shape([256,256,3])
    
    return tf_image


def mask_resolver(f, *argv, **kwargs):
    # get image
    tf_image = tf.image.decode_jpeg(f['mask'], channels=3)
        
    tf_image =  tf.image.resize_images(
        tf.image.convert_image_dtype(tf_image, tf.float32), [256,256])
    
    tf_image.set_shape([256,256,3])
    
    return tf_image



def ids_resolver(f, *argv, **kwargs):
    return f['id']

resolvers = {
    'inputs': uv_resolver,
    'images': image_resolver,
    'masks': mask_resolver,
    'ids': ids_resolver
}

provider = TFUV2ImageProvider(
    uv_data_path,
    img_data_path,
    resolvers=resolvers,
    batch_size=batch_size
)




def loss_network(data_eps, network_eps, alpha=1.0, collection=None):
    pass




def loss_discriminator(data_eps, network_eps, alpha=hps['disc_loss_weight']):
    _, states = network_eps
    logits_pred = states['discriminator_pred']
    logits_gt = states['discriminator_gt']

    discriminator_loss = tf.reduce_mean(-(tf.log(logits_gt + EPS) + tf.log(1 - logits_pred + EPS)))
    discriminator_loss *= alpha
    
    tf.losses.add_loss(discriminator_loss, loss_collection='discriminator_loss')
    tf.losses.add_loss(discriminator_loss)

    tf.summary.scalar('losses/discriminator', discriminator_loss)


def loss_generator(data_eps, network_eps, alpha=hps['gen_loss_weight'], l1_weight=hps['l1_loss_weight']):
    pred, states = network_eps
    
    n_channels = 3
    
    inputs = data_eps['inputs']
    targets = data_eps['images']
    masks = data_eps['masks']
    logits_pred = states['discriminator_pred']
    
    
    gen_loss_GAN = tf.reduce_mean(-tf.log(logits_pred + EPS))
    gen_loss_L1 = tf.reduce_mean(tf.abs(targets - pred) * masks)
    gen_loss = gen_loss_GAN + gen_loss_L1 * l1_weight
    gen_loss = gen_loss * alpha
    
    
    tf.losses.add_loss(gen_loss, loss_collection='generator_loss')
    tf.losses.add_loss(gen_loss)
    
    tf.summary.scalar('losses/generator', gen_loss)




def model_fn(inputs,
             n_channels=3,
             is_training=True,
             **kwargs):
    
    with slim.arg_scope([slim.batch_norm, slim.layers.dropout], is_training=is_training):
        states = {}

        # inputs
        targets = kwargs['data_eps']['images']

        # generators
        prediction = dm.models.gan.create_generator(
            inputs, n_channels, reuse=False, name="generator")

        # discriminators
        states['discriminator_gt'] = dm.models.gan.create_discriminator(
            inputs, targets, reuse=False, name='discriminator')
        states['discriminator_pred'] = dm.models.gan.create_discriminator(
            inputs, prediction, reuse=True, name='discriminator')

        return prediction, states




def custom_summary(data, endpoints, is_training=True):
    inputs = data['inputs']
    target = data['images']
    mask = data['masks']
    prediction, _ = endpoints
    
    prediction += 1.
    prediction /= 2.
    
    inputs += 1.
    inputs /= 2.
    inputs = tf.image.resize_images(inputs, [162, 256])
    
    tf.summary.image('resized/inputs', inputs)
    tf.summary.image('resized/prediction', prediction)
    tf.summary.image('resized/prediction/mask', prediction * mask)
    tf.summary.image('resized/target', target)
    tf.summary.image('resized/target/mask', target * mask)


# #### Prepare for Training



tf.reset_default_graph()
# create machine
# n_channels is the output channel number
uv2image = dm.DeepMachine(
    network_op=functools.partial(
        model_fn,
        n_channels=3
    )
)

# add losses
uv2image.add_loss_op(loss_network)
uv2image.add_loss_op(loss_generator)
uv2image.add_loss_op(loss_discriminator)

#add summary
uv2image.add_summary_op(dm.summary.summary_output_image_batch)
uv2image.add_summary_op(custom_summary)

# set ops
uv2image.train_op = dm.ops.train.gan




FLAGS.initial_learning_rate = hps['learning_rate']


if __name__ == "__main__":

    uv2image.train(
        train_data_op=functools.partial(provider.get, 'inputs', 'images', 'masks'),
        db_size=provider.size(),
        batch_size=batch_size,
        number_of_epochs=100,
        train_dir=FLAGS.train_dir
    )

