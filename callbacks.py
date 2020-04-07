import tensorflow as tf
import os


class CallBacks:

    def __init__(self, learning_rate=0.01, log_dir=None):
        super().__init__()
        self.log_dir = log_dir
        self.callbacks = self.get_callbacks()

    def _getTB(self):
        return tf.keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                              histogram_freq=100,
                                              write_graph=True,
                                              update_freq='epoch',
                                              write_images=False)

    def _getCP(self):
        return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(self.log_dir, 'cp-{epoch:04d}.ckpt'), verbose=1,
                                                  save_frequency=100)

    def get_callbacks(self):
        return [self._getTB(), self._getCP()]
