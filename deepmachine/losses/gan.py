import tensorflow as tf
slim = tf.contrib.slim

from . import helper
from .. import utils
from ..flags import FLAGS


def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.reduce_mean((in_ - target)**2)


def loss_discriminator(data_eps, network_eps):
    _, states = network_eps
    logits_pred = states['discriminator_pred']
    logits_gt = states['discriminator_gt']

    discriminator_gt_loss = mae_criterion(logits_gt, tf.ones_like(logits_gt))
    tf.losses.add_loss(discriminator_gt_loss,
                       loss_collection='discriminator_loss')
    tf.losses.add_loss(discriminator_gt_loss)

    discriminator_pred_loss = mae_criterion(
        logits_pred, tf.zeros_like(logits_pred))
    tf.losses.add_loss(discriminator_pred_loss,
                       loss_collection='discriminator_loss')
    tf.losses.add_loss(discriminator_pred_loss)

    tf.summary.scalar('losses/discriminator_gt', discriminator_gt_loss)
    tf.summary.scalar('losses/discriminator_pred', discriminator_pred_loss)

    tf.summary.scalar(
        'losses/discriminator',
        discriminator_gt_loss + discriminator_pred_loss)


def loss_generator(data_eps, network_eps):
    _, states = network_eps
    logits_pred = states['discriminator_pred']

    gen_loss = mae_criterion(logits_pred, tf.ones_like(logits_pred))

    tf.losses.add_loss(gen_loss, loss_collection='generator_loss')
    tf.losses.add_loss(gen_loss)
    tf.summary.scalar('losses/generator', gen_loss)


def loss_cyclegan_discriminator(data_eps, network_eps):

    _, states = network_eps

    DB_fake = states['disc_B_fake']
    DA_fake = states['disc_A_fake']
    DB_real = states['disc_B_real']
    DA_real = states['disc_A_real']

    db_loss_real = mae_criterion(DB_real, tf.ones_like(DB_real))
    db_loss_fake = mae_criterion(DB_fake, tf.zeros_like(DB_fake))
    db_loss = (db_loss_real + db_loss_fake) / 2.
    da_loss_real = mae_criterion(DA_real, tf.ones_like(DA_real))
    da_loss_fake = mae_criterion(DA_fake, tf.zeros_like(DA_fake))
    da_loss = (da_loss_real + da_loss_fake) / 2.

    tf.losses.add_loss(da_loss, loss_collection='discriminator_loss_A')
    tf.losses.add_loss(da_loss)
    tf.summary.scalar('losses/discriminator_A', da_loss)

    tf.losses.add_loss(db_loss, loss_collection='discriminator_loss_B')
    tf.losses.add_loss(db_loss)
    tf.summary.scalar('losses/discriminator_B', db_loss)


def loss_cyclegan_generator(data_eps, network_eps, n_channels=3, L1_lambda=10.):

    input_A = data_eps['inputs'][..., :n_channels]
    input_B = data_eps['inputs'][..., n_channels:]

    _, states = network_eps

    DB_fake = states['disc_B_fake']
    DA_fake = states['disc_A_fake']
    rec_A = states['rec_A']
    rec_B = states['rec_B']

    g_loss_a2b = mae_criterion(DB_fake, tf.ones_like(DB_fake)) + L1_lambda * abs_criterion(
        input_A, rec_A) + L1_lambda * abs_criterion(input_B, rec_B)

    g_loss_b2a = mae_criterion(DA_fake, tf.ones_like(DA_fake)) + L1_lambda * abs_criterion(
        input_A, rec_A) + L1_lambda * abs_criterion(input_B, rec_B)

    tf.losses.add_loss(g_loss_a2b, loss_collection='generator_loss_AB')
    tf.losses.add_loss(g_loss_a2b)
    tf.summary.scalar('losses/generator_AB', g_loss_a2b)

    tf.losses.add_loss(g_loss_b2a, loss_collection='generator_loss_BA')
    tf.losses.add_loss(g_loss_b2a)
    tf.summary.scalar('losses/generator_BA', g_loss_b2a)
