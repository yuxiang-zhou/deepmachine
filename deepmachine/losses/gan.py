import tensorflow as tf

from . import helper
from .. import utils

EPS = 1e-12


def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.reduce_mean((in_ - target)**2)

def cross_entropy(in_, target):
    return tf.reduce_mean(-(tf.log(in_ + EPS) + tf.log(1 - target + EPS)))


# pixtopix
def loss_discriminator(data_eps, network_eps, alpha=1.0):
    _, states = network_eps
    logits_pred = states['discriminator_pred']
    logits_gt = states['discriminator_gt']

    discriminator_loss = tf.reduce_mean(-(tf.log(logits_gt + EPS) + tf.log(1 - logits_pred + EPS)))
    discriminator_loss = discriminator_loss * alpha
    
    tf.losses.add_loss(discriminator_loss, loss_collection='discriminator_loss')
    tf.losses.add_loss(discriminator_loss)

    tf.summary.scalar('losses/discriminator_loss', discriminator_loss)


def loss_generator(data_eps, network_eps, alpha=1.0, l1_weight=100.):
    pred, states = network_eps
    
    n_channels = tf.shape(data_eps['inputs'])[-1] // 2
    
    inputs = data_eps['inputs'][..., :n_channels]
    targets = data_eps['inputs'][..., n_channels:]
    logits_pred = states['discriminator_pred']
    
    
    gen_loss_GAN = tf.reduce_mean(-tf.log(logits_pred + EPS))
    gen_loss_L1 = tf.reduce_mean(tf.abs(targets - pred))
    gen_loss = gen_loss_GAN + gen_loss_L1 * l1_weight
    gen_loss = gen_loss * alpha
    
    
    tf.losses.add_loss(gen_loss, loss_collection='generator_loss')
    tf.losses.add_loss(gen_loss)
    tf.summary.scalar('losses/generator', gen_loss)
    
    

    
# pose gan
def loss_posegan_generator(data_eps, network_eps, alpha=1.0, l1_weight=100.):
    pred, states = network_eps
    
    inputs = data_eps['inputs']
    logits_pred = states['discriminator_pred']
    
    gen_loss = mae_criterion(logits_pred, tf.ones_like(logits_pred))
    
    tf.losses.add_loss(gen_loss, loss_collection='generator_loss')
    tf.losses.add_loss(gen_loss)
    tf.summary.scalar('losses/generator', gen_loss)
    
def loss_posegan_discriminator(data_eps, network_eps, alpha=1.0):
    _, states = network_eps
    logits_pred = states['discriminator_pred']
    logits_gt = states['discriminator_gt']

    discriminator_loss = mae_criterion(logits_pred, tf.zeros_like(logits_pred))
    discriminator_loss += mae_criterion(logits_gt, tf.ones_like(logits_gt))
    discriminator_loss = discriminator_loss * alpha
    
    tf.losses.add_loss(discriminator_loss, loss_collection='discriminator_loss')
    tf.losses.add_loss(discriminator_loss)

    tf.summary.scalar('losses/discriminator_loss', discriminator_loss)
    

# LSTM GAN
def loss_lstm_discriminator(data_eps, network_eps, alpha=1.0):
    _, states = network_eps
    discriminator_loss = 0
    for logits_pred, logits_gt in zip(states['discriminator_pred'],states['discriminator_gt']):

        discriminator_loss += tf.reduce_mean(-(tf.log(logits_gt + EPS) + tf.log(1 - logits_pred + EPS)))
        
    discriminator_loss = discriminator_loss * alpha

    tf.losses.add_loss(discriminator_loss, loss_collection='discriminator_loss')
    tf.losses.add_loss(discriminator_loss)

    tf.summary.scalar('losses/discriminator_loss', discriminator_loss)


def loss_lstm_generator(data_eps, network_eps, alpha=1.0, l1_weight=100.):
    pred, states = network_eps
    
    n_channels = tf.shape(data_eps['inputs'])[-1] // 3
    
    inputs = data_eps['inputs'][:,0,:,: n_channels:]
    targets = data_eps['inputs'][:,0,:,:, :n_channels]
    logits_pred = states['discriminator_pred']
    
    
    gen_loss_GAN = tf.reduce_mean(-tf.log(logits_pred + EPS))
    gen_loss_L1 = tf.reduce_mean(tf.abs(targets - pred))
    gen_loss = gen_loss_GAN + gen_loss_L1 * l1_weight
    gen_loss = gen_loss * alpha
    
    
    tf.losses.add_loss(gen_loss, loss_collection='generator_loss')
    tf.losses.add_loss(gen_loss)
    tf.summary.scalar('losses/generator', gen_loss)
    


def loss_cyclegan_discriminator(data_eps, network_eps, alpha=1.0, disc_weight=1.0):

    _, states = network_eps

    # discriminator B losses
    DB_fake = states['disc_B_fake']
    DB_real = states['disc_B_real']
    

    db_loss_real = mae_criterion(DB_real, tf.ones_like(DB_real))
    db_loss_fake = mae_criterion(DB_fake, tf.zeros_like(DB_fake))
    db_loss = (db_loss_real + db_loss_fake) / 2.  * alpha * disc_weight
    
    tf.losses.add_loss(db_loss, loss_collection='discriminator_loss_B')
    tf.losses.add_loss(db_loss)
    tf.summary.scalar('losses/discriminator_B', db_loss)

    # discriminator A losses

    DA_fake = states['disc_A_fake']
    DA_real = states['disc_A_real']

    da_loss_real = mae_criterion(DA_real, tf.ones_like(DA_real))
    da_loss_fake = mae_criterion(DA_fake, tf.zeros_like(DA_fake))
    da_loss = (da_loss_real + da_loss_fake) / 2. * alpha

    tf.losses.add_loss(da_loss, loss_collection='discriminator_loss_A')
    tf.losses.add_loss(da_loss)
    tf.summary.scalar('losses/discriminator_A', da_loss)

    


def loss_cyclegan_generator(data_eps, network_eps, alpha=1.0, n_channels=3, L1_lambda=10., disc_weight=1.0):

    input_A = data_eps['inputs'][..., :n_channels]
    input_B = data_eps['inputs'][..., n_channels:]

    _, states = network_eps

    DB_fake = states['disc_B_fake']
    DA_fake = states['disc_A_fake']
    rec_A = states['rec_A']
    rec_B = states['rec_B']

    g_loss_a2b = mae_criterion(DB_fake, tf.ones_like(DB_fake)) * disc_weight + L1_lambda * abs_criterion(
        input_A, rec_A) + L1_lambda * abs_criterion(input_B, rec_B)
    g_loss_a2b = alpha * g_loss_a2b

    g_loss_b2a = mae_criterion(DA_fake, tf.ones_like(DA_fake)) * disc_weight + L1_lambda * abs_criterion(
        input_A, rec_A) + L1_lambda * abs_criterion(input_B, rec_B)
    g_loss_b2a = alpha * g_loss_b2a

    tf.losses.add_loss(g_loss_a2b, loss_collection='generator_loss_AB')
    tf.losses.add_loss(g_loss_a2b)
    tf.summary.scalar('losses/generator_AB', g_loss_a2b)

    tf.losses.add_loss(g_loss_b2a, loss_collection='generator_loss_BA')
    tf.losses.add_loss(g_loss_b2a)
    tf.summary.scalar('losses/generator_BA', g_loss_b2a)
