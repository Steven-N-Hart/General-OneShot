import tensorflow as tf


@tf.function
def triplet_loss(anchor=None,
                 positive=None,
                 negative=None):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, not used in this function.
    y_pred -- python list containing three objects:
            anchor:   the encodings for the anchor data
            positive: the encodings for the positive data (similar to anchor)
            negative: the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """

    # minimize the distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    # maximize distance between the anchor and the negative
    neg_dist = -tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

    return neg_dist, pos_dist


@tf.function
def lossless_triplet_loss(anchor=None,
                          positive=None,
                          negative=None, n=3, beta=None, epsilon=1e-8):
    """
    Implementation of the triplet loss function

    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    N  --  The number of dimension
    beta -- The scaling factor, N is recommended
    epsilon -- The Epsilon value to prevent ln(0)


    Returns:
    loss -- real number, value of the loss
    """
    if beta is None:
        beta = n
    # distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    # distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

    # Non Linear Values
    # -ln(-x/N+1)
    pos_dist = -tf.math.log(-tf.divide(pos_dist, beta) + 1 + epsilon)
    neg_dist = -tf.math.log(-tf.divide((n - neg_dist), beta) + 1 + epsilon)

    return neg_dist, pos_dist
