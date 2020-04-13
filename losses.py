import tensorflow as tf
import numpy as np

#@tf.function
def triplet_loss(y_batch_train, logits):
    """
    Implementation of the triplet loss function
    Arguments:
    y_batch_train -- true labels.
    logits -- predictions
    Returns:
    loss -- real number, value of the loss
    """
    #y_batch_train = tf.cast(tf.squeeze(y_batch_train), tf.float32)
    #logits = tf.cast(tf.squeeze(logits), tf.float32)
    loss = tf.keras.losses.binary_crossentropy(y_batch_train, logits)
    loss = tf.math.reduce_mean(loss)
    #loss = tf.math.reduce_sum(tf.math.abs(tf.math.subtract(tf.cast(y_batch_train, tf.float32), tf.squeeze(logits))))
    loss = tf.reshape(loss, (1,))
    return loss


