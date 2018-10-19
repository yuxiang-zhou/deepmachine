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
    LR_DECAY = 1 - FLAGS.lr_decay
    DB_PATH = FLAGS.dataset_path

    # local config
    W_DISC = 1
    GENERATOR_CH = 64
    DISCRIMINATOR_CH = 64
    DISC_THRED = 100.
    INPUT_SHAPE = 256
    DEPTH = 4
    DISC_SHAPE = INPUT_SHAPE//2**(DEPTH-1)
    valid = np.ones([BATCH_SIZE, DISC_SHAPE, DISC_SHAPE, 1], dtype=np.float32)
    fake = np.zeros([BATCH_SIZE, DISC_SHAPE, DISC_SHAPE, 1], dtype=np.float32)

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
                self.batch_size = len(self.images) if is_validation else batch_size
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

                return img_A, img_B

        train_A = ImageSequence('%s/train_A/' % DB_PATH)
        train_B = ImageSequence('%s/train_B/' % DB_PATH)
        train_generator = GeneratorSequence(train_A, train_B)
        
        return train_generator

    # ### Model
    def cyclegan_model():

        def build_generator(nf=GENERATOR_CH, depth=DEPTH, name=None, ks=4):
            inputs = dm.layers.Input(shape=[INPUT_SHAPE, INPUT_SHAPE, 3])
            # unet
            # outputs = dm.networks.UNet(inputs, [INPUT_SHAPE, INPUT_SHAPE, 3], nf=nf, ks=ks)
            # resnet
            outputs = dm.networks.ResNet50(inputs, [INPUT_SHAPE, INPUT_SHAPE, 3], nf=nf)
            # hourglass
            # outputs = dm.networks.Hourglass(inputs, [INPUT_SHAPE, INPUT_SHAPE, 3], nf=64, batch_norm='InstanceNormalization2D')
            return dm.Model(inputs, outputs, name=name)

        def build_discriminator(nf=DISCRIMINATOR_CH, depth=DEPTH, ks=4):
            inputs = dm.layers.Input(shape=[INPUT_SHAPE, INPUT_SHAPE, 3])
            validity = dm.networks.Discriminator(inputs, nf=nf, depth=DEPTH, ks=4)

            return dm.Model(inputs, validity)

        
        input_A = dm.layers.Input(shape=[INPUT_SHAPE, INPUT_SHAPE, 3])
        input_B = dm.layers.Input(shape=[INPUT_SHAPE, INPUT_SHAPE, 3])

        disc_A =  dm.DeepMachine(build_discriminator(), name="disc_A")
        disc_B = dm.DeepMachine(build_discriminator(), name="disc_B")

        optimizer_disc = dm.optimizers.Adam(LR*W_DISC, 0.5, decay=LR_DECAY)
        optimizer_gen = dm.optimizers.Adam(LR, 0.5, decay=LR_DECAY)

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
            loss_weights=[2, 2, 10., 10., 0., 0.],
        )

        return generator_model, generator_AB, generator_BA, disc_A, disc_B

    # ### Training
    def train_cyclegan_op(models, data, i_epoch, i_batch, epoch_end, training_history=None, **kwargs):
        generator_model, generator_AB, generator_BA, disc_A, disc_B = models
        imgs_A, imgs_B = next(data)
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
                result = disc.evaluate(in_y, true_y, verbose=0)
                disc_acc.append(result[1])

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
            }, {
                'inputs/input_A': imgs_A,
                'inputs/input_B': imgs_B,
                'fakes/fake_A': fake_A,
                'fakes/fake_B': fake_B,
                'recs/rec_A': rec_A,
                'recs/rec_B': rec_B,
            }
        )

    train_generator = build_data()
    train_queue = dm.utils.enqueue_generator(train_generator, workers=FLAGS.no_thread)
    models = cyclegan_model()

    history = dm.engine.training.train_monitor(
        models,
        train_queue, train_cyclegan_op,
        epochs=200, step_per_epoch=len(train_generator),
        callbacks=[
            train_generator
        ],
        verbose=FLAGS.verbose,
        logdir=LOGDIR,
    )


if __name__ == '__main__':
    main()
