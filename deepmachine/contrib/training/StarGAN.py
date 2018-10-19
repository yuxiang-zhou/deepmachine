# basic library
import os
import shutil
import math
import time
import menpo.io as mio
import menpo3d.io as m3io
import numpy as np
from pathlib import Path
from functools import partial

# deepmachine
import keras
import tensorflow as tf
import deepmachine as dm

# flag definitions
from deepmachine.flags import FLAGS


def main():
    LOGDIR = "{}/model_{}".format(FLAGS.logdir, time.time()
                                  ) if 'model_' not in FLAGS.logdir else FLAGS.logdir
    BATCH_SIZE = FLAGS.batch_size
    LR = FLAGS.lr
    LR_DECAY = 0  # 1 - FLAGS.lr_decay
    DB_PATH = FLAGS.dataset_path

    # local config
    W_DISC = 1
    GENERATOR_CH = 64
    DISCRIMINATOR_CH = 64
    DISC_THRED = 100.
    INPUT_SHAPE = 256
    DEPTH = 5
    N_CLASSES = 2
    DISC_SHAPE = INPUT_SHAPE//2**(DEPTH)
    valid = np.zeros([BATCH_SIZE*N_CLASSES*N_CLASSES, DISC_SHAPE, DISC_SHAPE, 1],
                    dtype=np.float32)
    fake = np.ones([BATCH_SIZE*N_CLASSES*N_CLASSES, DISC_SHAPE, DISC_SHAPE, 1],
                    dtype=np.float32)
    fake_label = -np.ones([BATCH_SIZE*N_CLASSES*N_CLASSES, N_CLASSES])
    # ### Dataset

    def build_data():
        def img_preprocessing(img):

            if img.n_channels == 1:
                img.pixels = np.tile(img.pixels, [3, 1, 1])

            offset = 64

            img = img.resize([INPUT_SHAPE + offset, INPUT_SHAPE + offset])

            # random crop
            offset_h, offset_w = map(int, np.random.sample([2]) * offset)
            img = img.crop([offset_h, offset_w], [offset_h +
                                                  INPUT_SHAPE, offset_w + INPUT_SHAPE])

            if np.random.rand() > 0.5:
                img = img.mirror()

            return img

        class ImageSequence(dm.utils.Sequence):

            def __init__(self, dirpath, batch_size=BATCH_SIZE, is_validation=False):
                self.images = list(Path(dirpath).rglob('*.*'))
                self.batch_size = len(
                    self.images) if is_validation else batch_size
                self.indexes = list(range(len(self.images)))
                np.random.shuffle(self.indexes)

            def on_epoch_end(self, *args, **kwargs):
                np.random.shuffle(self.indexes)

            def __len__(self):
                return math.floor(len(self.images) / self.batch_size)

            def __getitem__(self, idx):
                image_indexes = self.indexes[idx *
                                             self.batch_size:(idx + 1) * self.batch_size]

                batch_img = [img_preprocessing(mio.import_image(
                    self.images[i])) for i in image_indexes]
                batch_img = [img.pixels_with_channels_at_back() * 2. -
                             1 for img in batch_img]

                return np.array(batch_img)

        class GeneratorSequence(dm.utils.Sequence):
            def __init__(self, db_A, db_B):
                self.db_A = db_A
                self.db_B = db_B
                self.batch_size = np.min([
                    self.db_A.batch_size,
                    self.db_B.batch_size
                ])
                super().__init__()

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
                label_A = np.tile([[0, 1]], [len(img_A), 1])
                label_B = np.tile([[1, 0]], [len(img_B), 1])

                imgs = np.concatenate([img_A, img_B, img_A, img_B])
                target_labels = np.concatenate([label_B, label_A, label_A, label_B])
                original_labels = np.concatenate([label_A, label_B, label_A, label_B])

                return imgs, target_labels, original_labels

        train_A = ImageSequence('%s/train_A/' % DB_PATH, batch_size=BATCH_SIZE)
        train_B = ImageSequence('%s/train_B/' % DB_PATH, batch_size=BATCH_SIZE)
        train_generator = GeneratorSequence(train_A, train_B)

        return train_generator

    # ### Model
    def star_gan_model():

        def build_generator(nf=GENERATOR_CH, depth=DEPTH, name=None, ks=4):
            inputs = dm.layers.Input(shape=[INPUT_SHAPE, INPUT_SHAPE, 3])
            target_label = dm.layers.Input(shape=[N_CLASSES])
            target_label_conv = dm.layers.RepeatVector(
                INPUT_SHAPE * INPUT_SHAPE)(target_label)
            target_label_conv = dm.layers.Reshape(
                [INPUT_SHAPE, INPUT_SHAPE, N_CLASSES])(target_label_conv)
            merged_inputs = dm.layers.Concatenate()(
                [inputs, target_label_conv])
            # unet
            # outputs = dm.networks.UNet(inputs, [INPUT_SHAPE, INPUT_SHAPE, 3], nf=nf, ks=ks)
            # resnet
            outputs = dm.networks.ResNet50(
                merged_inputs, [INPUT_SHAPE, INPUT_SHAPE, 3], nf=nf, n_residule=6)
            # hourglass
            # outputs = dm.networks.Hourglass(inputs, [INPUT_SHAPE, INPUT_SHAPE, 3], nf=64, batch_norm='InstanceNormalization2D')
            return dm.Model([inputs, target_label], outputs, name=name)

        def build_discriminator(nf=DISCRIMINATOR_CH, depth=DEPTH, ks=4):
            inputs = dm.layers.Input(shape=[INPUT_SHAPE, INPUT_SHAPE, 3])
            validity, conv_feature = dm.networks.Discriminator(
                inputs, nf=nf, depth=depth, ks=4, return_endpoints=True)
            classes = dm.networks.conv2d(
                conv_feature, N_CLASSES, DISC_SHAPE, padding='valid', activation='softmax')
            classes = dm.layers.Reshape([N_CLASSES])(classes)

            return dm.Model(inputs, [validity, classes])

        def binary_crossentropy_none(y_true, y_pred):

            return dm.K.maximum(dm.K.mean(dm.K.binary_crossentropy(y_true, y_pred), axis=-1), 0)

        input_image = dm.layers.Input(shape=[INPUT_SHAPE, INPUT_SHAPE, 3])
        target_label = dm.layers.Input(shape=[N_CLASSES])
        original_label = dm.layers.Input(shape=[N_CLASSES])

        disc_model = dm.DeepMachine(build_discriminator(), name="disc")
        gen_optimizer = dm.optimizers.Adam(LR, 0.5, decay=LR_DECAY)
        dis_optimizer = dm.optimizers.Adam(LR, 0.5, decay=LR_DECAY)

        disc_model.compile(
            optimizer=dis_optimizer,
            loss=['mse', binary_crossentropy_none],
        )

        disc_model.trainable = False

        generator = build_generator(name="generator")

        fake_img = generator([input_image, target_label])
        rec_img = generator([fake_img, original_label])
        valid_img, fake_classes = disc_model(fake_img)

        generator_model = dm.DeepMachine(
            inputs=[input_image, target_label, original_label],
            outputs=[
                rec_img, valid_img, fake_classes
            ]
        )

        generator_model.compile(
            optimizer=gen_optimizer,
            loss=['mae', 'mse', 'categorical_crossentropy'],
            loss_weights=[10., 1., 1.],
        )

        return generator_model, generator, disc_model

    # ### Training
    def train_op(models, data, i_epoch, i_batch, epoch_end, training_history=None, **kwargs):
        generator_model, generator, disc_model = models
        imgs, target_labels, original_labels = dm.engine.training.generator_adapter(
            data)
        sess = dm.K.get_session()
        # ----------------------
        #  Train Discriminators
        # ----------------------

        # Translate images to opposite domain
        fake_imgs = generator.predict([imgs, target_labels])

        d_loss_real = disc_model.train_on_batch(imgs, [valid, original_labels])
        d_loss_fake = disc_model.train_on_batch(fake_imgs, [fake, fake_label])
        d_loss = np.add(d_loss_real, d_loss_fake)

        # ------------------
        #  Train Generators
        # ------------------
        # Train the generators
        g_loss = generator_model.train_on_batch(
            [imgs, target_labels, original_labels],
            [imgs, valid, target_labels]
        )

        logs = dm.utils.Summary(
            {
                "losses/G_loss": g_loss[0],
                "losses/G_recon": np.mean(g_loss[1]),
                "losses/G_adv": np.mean(g_loss[2]),
                "losses/G_cls": np.mean(g_loss[3]),
                "losses/D_loss": d_loss[0],
                "losses/D_adv": np.mean(d_loss[1]),
                "losses/D_cls": np.mean(d_loss[2]),
                "learning_rate": generator_model.optimizer.lr.eval(sess),
            }
        )

        if epoch_end:
            rec_imgs = generator.predict([fake_imgs, original_labels])
            logs.update_images({
                'inputs/input': imgs,
                'fakes/fake': fake_imgs,
                'recs/rec': rec_imgs
            })

        return logs

    train_generator = build_data()
    train_queue = dm.utils.enqueue_generator(
        train_generator, workers=FLAGS.no_thread)
    models = star_gan_model()
    dis_lr_decay = dm.callbacks.LearningRateScheduler(
        schedule=lambda epoch: LR * FLAGS.lr_decay ** epoch)
    dis_lr_decay.set_model(models[-1])

    history = dm.engine.training.train_monitor(
        models,
        train_queue, train_op,
        epochs=200, step_per_epoch=len(train_generator),
        callbacks=[
            train_generator,
            dis_lr_decay
        ],
        verbose=FLAGS.verbose,
        logdir=LOGDIR,
    )


if __name__ == '__main__':
    main()
