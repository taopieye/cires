import tensorflow as tf 
from tensorflow.keras import losses, layers
 

#helpers
class Sampling(layers.Layer):
    #uses z_mean, z_log_var to sample z, encoding vector
    
    def call(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch,dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def kl_divergence(z_mean, z_log_var):
    kl_loss = -0.5*(1+z_log_var-tf.square(z_mean)-tf.exp(z_log_var))
    return tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

def reconstruction_loss(real, reconstruction):
    return tf.reduce_mean(tf.reduce_sum(
        losses.binary_crossentropy(real, reconstruction), axis=(1,2)))