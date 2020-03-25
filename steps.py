import tensorflow as tf

def write_tb(writer, step, neg_dist, pos_dist, total_dist, siamese_net, num_layers=5):
    with writer.as_default():
        tf.summary.scalar('neg_dist', neg_dist, step=step)
        tf.summary.scalar('pos_dist', pos_dist, step=step)
        tf.summary.scalar('total_dist', total_dist, step=step)

        #for i, layer in enumerate(siamese_net.get_weights()[-num_layers:]):
        #    tf.summary.histogram('layer_{0}'.format(i), layer, step=step)