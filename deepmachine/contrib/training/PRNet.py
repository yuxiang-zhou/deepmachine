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
tf.app.flags.DEFINE_boolean('use_ae', False, '''Use AutoEncoder as Constrain''')
tf.app.flags.DEFINE_string('ae_path', '', '''Use AutoEncoder as Constrain''')
from deepmachine.flags import FLAGS

def main():

    BATCH_SIZE = FLAGS.batch_size
    INPUT_SHAPE = 256
    LR = FLAGS.lr
    DATA_PATH = FLAGS.dataset_path
    LOGDIR = "{}/model_{}".format(FLAGS.logdir, time.time()
                                  ) if 'model_' not in FLAGS.logdir else FLAGS.logdir

    weight_mask = np.load('/vol/phoebe/yz4009/notebooks/Projects/MLProjects/data/face_uv_weight_mask.npy')[None,...,None]

    # Dataset

    def build_data():

        dataset = dm.data.provider.TFRecordNoFlipProvider(
            DATA_PATH,
            dm.utils.union_dict([
                dm.data.provider.features.image_feature,
                dm.data.provider.features.uvxyz_feature
            ]),
            augmentation=False,
            resolvers={
                'images': dm.data.provider.resolvers.image_resolver,
                'uvxyz': dm.data.provider.resolvers.uvxyz_resolver
            }
        )
        dataset = dm.data.provider.DatasetQueue(
            dataset, n_proccess=FLAGS.no_thread, batch_size=BATCH_SIZE)
        tf_data = dataset.get('images', 'uvxyz')

        train_x = [tf_data['images']]
        train_y = [tf_data['uvxyz']] * 2

        if FLAGS.use_ae and FLAGS.ae_path:
            ae_dataset = dm.data.provider.TFRecordNoFlipProvider(
                FLAGS.ae_path,
                dm.utils.union_dict([
                    dm.data.provider.features.uvxyz_feature
                ]),
                augmentation=False,
                resolvers={
                    'uvxyz': dm.data.provider.resolvers.uvxyz_resolver,
                    'dummy': dm.data.provider.resolvers.dummy_resolver,
                }
            )
            ae_dataset = dm.data.provider.DatasetQueue(
                ae_dataset, n_proccess=FLAGS.no_thread, batch_size=BATCH_SIZE)
            tf_ae_data = ae_dataset.get('uvxyz','dummy')
            train_y.append(tf_ae_data['uvxyz'])

        return train_x, train_y

    def model_builder():
        optimizer = dm.optimizers.Adam(lr=LR, clipnorm=1., decay=0.)

        if FLAGS.use_ae and FLAGS.ae_path:

            # encoder
            ae_input = dm.layers.Input(shape=[INPUT_SHAPE, INPUT_SHAPE, 3], name='ae_input')
            embeding_output = dm.networks.Encoder2D(ae_input, 128, depth=8, nf=32, batch_norm=False)
            encoder_model = dm.Model(inputs=[ae_input], outputs=[embeding_output], name='encoder_model')

            # decoder
            input_embeding = dm.layers.Input(shape=[128,], name='ae_input_embeding')
            ae_output = dm.networks.Decoder2D(input_embeding, [INPUT_SHAPE, INPUT_SHAPE, 3], depth=8, nf=32, batch_norm=False)
            decoder_model = dm.Model(inputs=[input_embeding], outputs=[ae_output], name='decoder_model')
            
            # combined model
            ae_model = dm.DeepMachine(
                inputs=[ae_input],
                outputs=[decoder_model(encoder_model(ae_input))]
            )
            ae_model.compile(
                optimizer=optimizer,
                loss=['mae']
            )
            ae_model.trainable = False

        input_image = dm.layers.Input(
            shape=[INPUT_SHAPE, INPUT_SHAPE, 3], name='input_image')
        uvxyz_prediction = dm.networks.Hourglass(
            input_image, [256, 256, 3], nf=64, batch_norm='InstanceNormalization2D')

        merged_input = dm.layers.Concatenate()([input_image, uvxyz_prediction])

        uvxyz_prediction_refine = dm.networks.Hourglass(
            merged_input, [256, 256, 3], nf=64, batch_norm='InstanceNormalization2D')

        outputs = [uvxyz_prediction, uvxyz_prediction_refine]
        if FLAGS.use_ae and FLAGS.ae_path:
            uvxyz_prediction_ae = encoder_model(uvxyz_prediction_refine)
            outputs.append(uvxyz_prediction_ae)
        
        train_model = dm.DeepMachine(
            inputs=input_image, outputs=outputs)

        

        def weighted_uv_loss(y_true, y_pred):

            loss = dm.K.mean(weight_mask * dm.K.abs(y_true - y_pred))

            return loss

        train_model.compile(
            optimizer=optimizer,
            loss=[
                weighted_uv_loss,
                weighted_uv_loss,
                'mae'
            ],
            loss_weights=[1, 1, 1]
        )

        if FLAGS.use_ae and FLAGS.ae_path:
            return train_model, ae_model, encoder_model, decoder_model
        else:
            return train_model

    ### Training
    if FLAGS.use_ae and FLAGS.ae_path:
        datas = build_data()
        models = model_builder()

        def train_op(models, data, i_epoch, i_batch, epoch_end, training_history=None, **kwargs):
            prnet_model, ae_model, encoder_model, decoder_model = models
            sess = dm.K.get_session()
            train_x, train_y = dm.engine.training.tf_dataset_adapter(data, i_epoch, i_batch, epoch_end)
            
            prnet_data = train_y[:-1]
            ae_data = train_y[-1]
            # ----------------------
            #  Train AutoEncoder
            # ----------------------
            ae_loss = ae_model.train_on_batch(ae_data, ae_data)

            # ------------------
            #  Train PRNet
            # ------------------
            # Train the generators
            embedding = encoder_model.predict(prnet_data[-1])
            prnet_loss = prnet_model.train_on_batch(
                train_x,
                prnet_data + [embedding]
            )

            logs = dm.utils.Summary(
                {
                    "losses/AE_loss": ae_loss,
                    "losses/PRNet_total_loss": prnet_loss[0],
                    "losses/PRNet_loss_0": prnet_loss[1],
                    "losses/PRNet_loss_1": prnet_loss[2],
                    "losses/PRNet_loss_2": prnet_loss[3],
                    "learning_rate": prnet_model.optimizer.lr.eval(sess)
                }
            )

            if epoch_end:
                predict_ae = ae_model.predict(ae_data)
                predict_prnet = prnet_model.predict(train_x)

                logs.update_images({
                    'inputs/prnet': train_x[0],
                    'inputs/ae': ae_data,

                    'outputs/prnet_0': predict_prnet[0],
                    'outputs/prnet_1': predict_prnet[1],
                    'outputs/prnet_2': predict_prnet[2],
                    'outputs/ae': predict_ae,

                    'targets/prnet': train_y[-1],
                    'targets/ae': ae_data,
                })

            return logs
        
        lr_scheduler = dm.callbacks.LearningRateScheduler(
                schedule=lambda epoch: LR * FLAGS.lr_decay ** epoch)
        lr_scheduler.set_model(models[0])

        history = dm.engine.training.train_monitor(
            models,
            datas, 
            train_op,
            epochs=200, 
            step_per_epoch=15000 // BATCH_SIZE,
            verbose=FLAGS.verbose,
            logdir=LOGDIR,
            callbacks=[
                lr_scheduler
            ]
        )
    else:
        model_builder().fit(
            build_data(),
            epochs=200, 
            step_per_epoch=15000 // BATCH_SIZE,
            logdir=LOGDIR,
            verbose=2,
            summary_ops=[],
            lr_decay=0.99,
        )


if __name__ == '__main__':
    main()
