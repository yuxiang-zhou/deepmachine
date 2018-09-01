import shutil, math, datetime, time
import menpo.io as mio
import menpo3d.io as m3io
import numpy as np
import deepmachine as dm
import tensorflow as tf
import keras
from deepmachine.flags import FLAGS
from matplotlib import pyplot as plt
from functools import partial
from deepmachine import data_provider
from menpo.shape import PointCloud, TriMesh, ColouredTriMesh
from pathlib import Path
from menpo.visualize import print_progress, print_dynamic
from menpo.image import Image
from menpo.transform import Translation
from menpo3d.camera import PerspectiveCamera
from menpo3d.unwrap import optimal_cylindrical_unwrap
from menpo3d.rasterize import rasterize_mesh

def main():
    LOGDIR="{}/model_{}".format(FLAGS.logdir, time.time()) if 'model_' not in FLAGS.logdir else FLAGS.logdir
    BATCH_SIZE = FLAGS.batch_size
    LR = FLAGS.lr
    DB_PATH = FLAGS.dataset_path

    # local config
    W_DISC = 1
    GENERATOR_CH = 64
    DISCRIMINATOR_CH = 64
    DISC_THRED = 80.
    INPUT_SHAPE = 128
    DISC_SHAPE = INPUT_SHAPE//2**4
    valid = np.ones([BATCH_SIZE,DISC_SHAPE,DISC_SHAPE,1], dtype=np.float32)
    fake = np.zeros([BATCH_SIZE,DISC_SHAPE,DISC_SHAPE,1], dtype=np.float32)


    # ### Dataset
    def img_preprocessing(img):
        
        if img.n_channels == 1:
            img.pixels = np.tile(img.pixels, [3,1,1])
            
        offset = 64
            
        img = img.resize([INPUT_SHAPE + offset, INPUT_SHAPE + offset])
        
        # random crop
        offset_h, offset_w = map(int, np.random.sample([2]) * offset)
        img = img.crop([offset_h, offset_w],[offset_h + INPUT_SHAPE, offset_w + INPUT_SHAPE])
        
        if np.random.rand() > 0.5:
            img = img.mirror()
            
        return img

    class ImageSequence(dm.utils.Sequence):

        def __init__(self, dirpath, batch_size=BATCH_SIZE, is_validation=False):
            self.images = list(Path(dirpath).rglob('*.*'))
            self.batch_size = len(self.images) if is_validation else batch_size
            self.indexes = list(range(len(self.images)))
            np.random.shuffle(self.indexes)
            
        def on_epoch_end(self, *args, **kwargs):
            np.random.shuffle(self.indexes)

        def __len__(self):
            return math.floor(len(self.images) / self.batch_size)

        def __getitem__(self, idx):
            image_indexes = self.indexes[idx * self.batch_size:(idx + 1) *self.batch_size]
            
            batch_img = [img_preprocessing(mio.import_image(self.images[i])) for i in image_indexes]
            batch_img = [img.pixels_with_channels_at_back() * 2. - 1 for img in batch_img]

            return np.array(batch_img)

        
    class GeneratorSequence(dm.utils.Sequence):
        def __init__(self, db_A, db_B):
            self.db_A = db_A
            self.db_B = db_B
            self.batch_size = np.min([
                self.db_A.batch_size,
                self.db_B.batch_size
            ])
            
        def on_epoch_end(self, *args, **kwargs):
            self.db_A.on_epoch_end()
            self.db_B.on_epoch_end()
            
        def __len__(self):
            return np.min([
                len(self.db_A), len(self.db_B)
            ])
        
        def __getitem__(self, idx):
            img_A = self.db_A[idx]
            img_B = self.db_B[idx]
            
            return img_A, img_B

    # ### Model
    def cyclegan_model():

        def build_generator_new(init_channels=32, depth=4, name=None):
            inputs = dm.layers.Input(shape=[INPUT_SHAPE,INPUT_SHAPE,3])
            net = inputs
            for i in range(depth):
                net = dm.networks.Encoding2D(net, init_channels * 2 ** i, use_coordconv=False, batch_norm=True)

            for i in range(depth-1):
                net = dm.networks.Decoding2D(net, init_channels * 2 ** (depth-2-i), batch_norm=True)

            net = dm.networks.Decoding2D(net, 3, batch_norm=True)
            outputs = dm.networks.conv2d(net, filters=3, kernel_size=4, strides=1, padding='same', activation='tanh')
            return dm.Model(inputs, outputs, name=name)

        def build_discriminator_new(init_channels=32, depth=4):
            inputs = dm.layers.Input(shape=[INPUT_SHAPE,INPUT_SHAPE,3])
            net = inputs
            for i in range(depth):
                net = dm.networks.Encoding2D(net, init_channels * 2 ** i, batch_norm=True)

            validity = dm.layers.Conv2D(1, kernel_size=4, strides=1, padding='same')(net)

            return dm.Model(inputs, validity)
        
        def build_generator(init_channels=GENERATOR_CH, depth=4, name=None):
            """U-Net Generator"""

            def conv2d(layer_input, filters, f_size=4):
                """Layers used during downsampling"""
                d = dm.layers.Conv2D(filters, kernel_size=f_size, strides=2, padding='same', activation='relu')(layer_input)
                d = dm.layers.InstanceNormalization2D()(d)
                return d

            def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
                """Layers used during upsampling"""
                u = dm.layers.UpSampling2D(size=2)(layer_input)
                u = dm.layers.Conv2D(filters, kernel_size=f_size, padding='same', activation='relu')(u)
                if dropout_rate:
                    u = dm.layers.Dropout(dropout_rate)(u)
                u = dm.layers.InstanceNormalization2D()(u)
                u = dm.layers.Concatenate()([u, skip_input])
                return u

            # Image input
            d0 = dm.layers.Input(shape=[INPUT_SHAPE,INPUT_SHAPE,3])

            # Downsampling
            d1 = conv2d(d0, init_channels)
            d2 = conv2d(d1, init_channels*2)
            d3 = conv2d(d2, init_channels*4)
            d4 = conv2d(d3, init_channels*8)

            # Upsampling
            u1 = deconv2d(d4, d3, init_channels*4)
            u2 = deconv2d(u1, d2, init_channels*2)
            u3 = deconv2d(u2, d1, init_channels)

            u4 = dm.layers.UpSampling2D(size=2)(u3)
            output_img = dm.layers.Conv2D(3, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

            return dm.Model(d0, output_img)
        
        def build_discriminator(init_channels=DISCRIMINATOR_CH, depth=4):

            def d_layer(layer_input, filters, f_size=4, normalization=True):
                """Discriminator layer"""
                d = dm.layers.Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
                d = dm.layers.LeakyReLU(alpha=0.2)(d)
                if normalization:
                    d = dm.layers.InstanceNormalization2D()(d)
                return d

            img = dm.layers.Input(shape=[INPUT_SHAPE,INPUT_SHAPE,3])

            d1 = d_layer(img, init_channels, normalization=False)
            d2 = d_layer(d1, init_channels*2)
            d3 = d_layer(d2, init_channels*4)
            d4 = d_layer(d3, init_channels*8)

            validity = dm.layers.Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

            return dm.Model(img, validity)


        input_A = dm.layers.Input(shape=[INPUT_SHAPE,INPUT_SHAPE,3])
        input_B = dm.layers.Input(shape=[INPUT_SHAPE,INPUT_SHAPE,3])

        disc_A = dm.DeepMachine(build_discriminator(), name="disc_A")
        disc_B = dm.DeepMachine(build_discriminator(), name="disc_B")
        
        optimizer_disc = dm.optimizers.Adam(LR*W_DISC, 0.5)
        optimizer_gen = dm.optimizers.Adam(LR, 0.5)
        
        disc_A.compile(
            optimizer=optimizer_disc,
            loss=['mse'],
            metrics=['accuracy'],
        )
        disc_B.compile(
            optimizer=optimizer_disc,
            loss=['mse'],
            metrics=['accuracy'],
        )
        
        disc_A.trainable = False
        disc_B.trainable = False

        generator_AB = build_generator(name="generator_AB")
        generator_BA = build_generator(name="generator_BA")

        fake_A = generator_BA(input_B)
        fake_B = generator_AB(input_A)

        rec_A = generator_BA(fake_B)
        rec_B = generator_AB(fake_A)

        id_A = generator_BA(input_A)
        id_B = generator_AB(input_B)

        
        valid_A = disc_A(fake_A)
        valid_B = disc_B(fake_B)
    
        generator_model = dm.DeepMachine(
            inputs=[input_A, input_B],
            outputs=[
                valid_A, valid_B,
                rec_A, rec_B,
                id_A, id_B
            ]
        )
        
        generator_model.compile(
            optimizer=optimizer_gen,
            loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
            loss_weights=[1, 1, 10.,10.,0.,0.],
        )
        
        
        return generator_model, generator_AB, generator_BA, disc_A, disc_B


    val_A = ImageSequence('%s/test_A/'%DB_PATH, is_validation=True)
    val_B = ImageSequence('%s/test_B/'%DB_PATH, is_validation=True)
    val_generator = GeneratorSequence(val_A, val_B)
    train_A = ImageSequence('%s/train_A/'%DB_PATH)
    train_B = ImageSequence('%s/train_B/'%DB_PATH)
    train_generator = GeneratorSequence(train_A, train_B)

    models = cyclegan_model()

    # ### Training

    def train_cyclegan_op(models, data, i_epoch, i_batch, epoch_end, training_history=None, **kwargs):
        generator_model, generator_AB, generator_BA, disc_A, disc_B = models
        imgs_A, imgs_B = data[i_batch]
        sess = dm.K.get_session()
        # ----------------------
        #  Train Discriminators
        # ----------------------

        # Translate images to opposite domain
        fake_B = generator_AB.predict(imgs_A)
        fake_A = generator_BA.predict(imgs_B)
        rec_B = generator_AB.predict(fake_A)
        rec_A = generator_BA.predict(fake_B)

        # Train the discriminators (original images = real / translated = Fake)
        
        if training_history and 'losses/acc' in training_history.history and training_history.history['losses/acc'][-1] > DISC_THRED:
            disc_acc = []
            for disc, true_y, in_y in [
                [disc_A, valid, imgs_A],
                [disc_A, fake, fake_A],
                [disc_B, valid, imgs_B],
                [disc_B, fake, fake_B],
            ]:
                pred_y = disc.predict(in_y)
                disc_acc.append(
                    np.mean(sess.run(keras.metrics.binary_accuracy(true_y, pred_y)))
                )
            
            d_loss = [
                training_history.history['losses/D_loss'][-1],
                np.mean(disc_acc)
            ]
        else:
            dA_loss_real = disc_A.train_on_batch(imgs_A, valid)
            dA_loss_fake = disc_A.train_on_batch(fake_A, fake)
            dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

            dB_loss_real = disc_B.train_on_batch(imgs_B, valid)
            dB_loss_fake = disc_B.train_on_batch(fake_B, fake)
            dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

            # Total disciminator loss
            d_loss = 0.5 * np.add(dA_loss, dB_loss)

        # ------------------
        #  Train Generators
        # ------------------
        # Train the generators
        g_loss = generator_model.train_on_batch(
            [imgs_A, imgs_B], 
            [valid, valid, imgs_A, imgs_B, imgs_A, imgs_B]
        )
        
        return dm.utils.Summary(
            {
                "losses/D_loss": d_loss[0], 
                "losses/acc":  100*d_loss[1],
                "losses/G_loss": g_loss[0], 
                "losses/adv": np.mean(g_loss[1:3]), 
                "losses/recon": np.mean(g_loss[3:5]), 
                "losses/id": np.mean(g_loss[5:6]),
                "learning_rate": generator_model.optimizer.lr.eval(sess)
            },{
                'inputs/input_A': imgs_A,
                'inputs/input_B': imgs_B,
                'fakes/fake_A': fake_A,
                'fakes/fake_B': fake_B,
                'recs/rec_A': rec_A,
                'recs/rec_B': rec_B,
            }
        )

    history = dm.engine.training.train_monitor(
        models, 
        train_generator, train_cyclegan_op, 
        epochs=200, step_per_epoch=len(train_generator), 
        callbacks=[
            dm.callbacks.LambdaCallback(on_epoch_end=lambda x,y: train_generator.on_epoch_end())
        ],
        verbose=1,
        logdir=LOGDIR,
    )

if __name__ == '__main__':
    main()