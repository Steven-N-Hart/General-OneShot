from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import logging
import os
import sys
import tensorflow as tf
from model_factory import GetModel
from losses import triplet_loss as loss_fn
from preprocess import get_doublets_and_labels, preprocess
import numpy as np

#os.environ['CUDA_VISIBLE_DEVICES']="2,3"
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
if len(tf.config.experimental.list_physical_devices('GPU')) == 0:
    exit()
for g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(g, True)

tf.config.set_soft_device_placement(True)
#tf.debugging.set_log_device_placement(True)
###############################################################################
# Input Arguments
###############################################################################

parser = argparse.ArgumentParser(description='Run a Siamese Network with a triplet loss on a folder of images.')
parser.add_argument("-t", "--image_dir_train",
                    dest='image_dir_train',
                    required=True,
                    help="File path ending in folders that are to be used for model training")

parser.add_argument("-v", "--image_dir_validation",
                    dest='image_dir_validation',
                    default=None,
                    help="File path ending in folders that are to be used for model validation")

parser.add_argument("-m", "--model-name",
                    dest='model_name',
                    default='custom',
                    choices=['custom',
                             'DenseNet121',
                             'DenseNet169',
                             'DenseNet201',
                             'InceptionResNetV2',
                             'InceptionV3',
                             'MobileNet',
                             'MobileNetV2',
                             'NASNetLarge',
                             'NASNetMobile',
                             'ResNet50',
                             'VGG16',
                             'VGG19',
                             'Xception'],
                    help="Models available from tf.keras")

parser.add_argument("-o", "--optimizer-name",
                    dest='optimizer',
                    default='Adam',
                    choices=['Adadelta',
                             'Adagrad',
                             'Adam',
                             'Adamax',
                             'Ftrl',
                             'Nadam',
                             'RMSprop',
                             'SGD'],
                    help="Optimizers from tf.keras")

parser.add_argument("-p", "--patch_size",
                    dest='patch_size',
                    help="Patch size to use for training",
                    default=256, type=int)

parser.add_argument("-c", "--embedding_size",
                    dest='embedding_size',
                    help="How large should the embedding dimension be",
                    default=128, type=int)

parser.add_argument("-l", "--log_dir",
                    dest='log_dir',
                    default='log_dir',
                    help="Place to store the tensorboard logs")

parser.add_argument("-r", "--learning-rate",
                    dest='lr',
                    help="Learning rate",
                    default=0.01, type=float)

parser.add_argument("-e", "--num-epochs",
                    dest='num_epochs',
                    help="Number of epochs to use for training",
                    default=15, type=int)

parser.add_argument("-b", "--batch-size",
                    dest='BATCH_SIZE',
                    help="Number of batches to use for training",
                    default=10, type=int)

parser.add_argument("-V", "--verbose",
                    dest="logLevel",
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                    default="DEBUG",
                    help="Set the logging level")

parser.add_argument("-F", "--filetype",
                    dest="filetype",
                    choices=['tfrecords', 'images'],
                    default="images",
                    help="Set the logging level")

parser.add_argument("--tfrecord_image",
                    dest="tfrecord_image",
                    default="image/encoded",
                    help="Set the logging level")

parser.add_argument("--tfrecord_label",
                    dest="tfrecord_label",
                    default="null",
                    help="Set the logging level")

parser.add_argument('-f', "--log_freq",
                    dest="log_freq",
                    default=100,
                    help="Set the logging frequency for saving Tensorboard updates", type=int)

parser.add_argument('-a', "--accuracy_num_batch",
                    dest="acc_num_batch",
                    default=20,
                    help="Number of batches to consider to calculate training and validation accuracy", type=int)
                    
args = parser.parse_args()

logging.basicConfig(stream=sys.stderr, level=args.logLevel,
                    format='%(name)s (%(levelname)s): %(message)s')

logger = logging.getLogger(__name__)
logger.setLevel(args.logLevel)

###############################################################################
# Set some globals
###############################################################################
out_dir = os.path.join(args.log_dir,
                       args.model_name + '_' + args.optimizer + '_' + str(args.lr) + '_' + str(args.BATCH_SIZE))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

checkpoint_name = 'training_checkpoints'

###############################################################################
# Begin priming the data generation pipeline
###############################################################################
label_dict ={'positive': 1, 'negative': 0}
anchor, other, labels = get_doublets_and_labels(args.image_dir_train, label_dict=label_dict)
ds = tf.data.Dataset.from_tensor_slices(({"anchor":anchor, "other": other}, labels))
ds = ds.map(preprocess).batch(args.BATCH_SIZE, drop_remainder=True)  # Convert filepaths to images and label strings to ints

v_ds = args.image_dir_validation
if v_ds is not None:
    v_anchor, v_other, v_labels = get_doublets_and_labels(args.image_dir_validation, label_dict=label_dict)
    v_ds = tf.data.Dataset.from_tensor_slices(({"anchor": v_anchor, "other": v_other}, v_labels))
    v_ds = v_ds.map(preprocess).batch(args.BATCH_SIZE,
                                  drop_remainder=True)  # Convert filepaths to images and label strings to ints

###############################################################################
# Define callbacks
###############################################################################
def scheduler(epoch):
    if epoch < 5:
        return 0.01
    elif epoch < 10:
        return 0.001
    else:
        return 0.001 * tf.math.exp(0.1 * (10 - epoch))


cb = [tf.keras.callbacks.TensorBoard(log_dir=out_dir, histogram_freq=1, update_freq=args.log_freq),
      tf.keras.callbacks.ModelCheckpoint(os.path.join(out_dir, 'siamesenet'), monitor='mse', verbose=0,
                                         mode='auto'),
        tf.keras.callbacks.LearningRateScheduler(scheduler)
      ]

###############################################################################
# Build model
###############################################################################
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    m = GetModel(model_name=args.model_name, img_size=args.patch_size, embedding_size=args.embedding_size)
    model = m.build_model()
    model.summary()
    optimizer = m.get_optimizer(args.optimizer, lr=args.lr)
    model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['binary_accuracy', 'mse', tf.keras.metrics.AUC()])

if args.image_dir_validation is None:
    model.fit(ds, epochs=args.num_epochs, callbacks=cb )
else:
    model.fit(ds, epochs=args.num_epochs, callbacks=cb, validation_data=v_ds)

outfile_dir = os.path.join(out_dir,'siamesenet')
model.reset_metrics()
model.save(outfile_dir, save_format='tf')
print('Completed and saved {outfile_dir}')
exit(0)


