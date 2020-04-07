from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import logging
import os
import sys
import tensorflow as tf
from model_factory import GetModel
import re
from preprocess import get_doublets_and_labels, preprocess

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
                    default=5, type=int)

parser.add_argument("-b", "--batch-size",
                    dest='BATCH_SIZE',
                    help="Number of batches to use for training",
                    default=1, type=int)

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
                       args.model_name + '_' + args.optimizer + '_' + str(args.lr))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

checkpoint_name = 'training_checkpoints'

###############################################################################
# Begin priming the data generation pipeline
###############################################################################
label_dict ={'positive': 1, 'negative': 0}
anchor, other, labels = get_doublets_and_labels(args.image_dir_train, label_dict=label_dict)
#ds = tf.data.Dataset.from_tensor_slices((ds, labels))
ds = tf.data.Dataset.from_tensor_slices(({"anchor":anchor, "other": other}, labels))
# Convert filepaths to images and label strings to ints
ds = ds.map(preprocess).batch(5, drop_remainder=True)

# ####################################################################
# Temporary cleaning function
# ####################################################################
overwrite = True
if overwrite is True:
    for root, dirs, files in os.walk(out_dir):
        for file in filter(lambda x: re.match(checkpoint_name, x), files):
            print('Removing: {}'.format(os.path.join(root, file)))
            os.remove(os.path.join(root, file))
        for file in filter(lambda x: re.match('checkpoint', x), files):
            print('Removing: {}'.format(os.path.join(root, file)))
            os.remove(os.path.join(root, file))
        for file in filter(lambda x: re.match('events', x), files):
            print('Removing: {}'.format(os.path.join(root, file)))
            os.remove(os.path.join(root, file))
        for file in filter(lambda x: re.match('ckpt', x), files):
            print('Removing: {}'.format(os.path.join(root, file)))
            os.remove(os.path.join(root, file))
        for file in filter(lambda x: re.match('siamese', x), files):
            print('Removing: {}'.format(os.path.join(root, file)))
            os.remove(os.path.join(root, file))

###############################################################################
# Define callbacks
###############################################################################
#cb = CallBacks(learning_rate=args.lr, log_dir=out_dir)
cb=[tf.keras.callbacks.TensorBoard(log_dir=out_dir, write_graph=False, update_freq=100),
    tf.keras.callbacks.ModelCheckpoint(os.path.join(out_dir,'siamesenet'), monitor='loss', verbose=0, mode='auto'),
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
]

###############################################################################
# Build model
###############################################################################
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    m = GetModel(model_name=args.model_name, img_size=args.patch_size, embedding_size=args.embedding_size)
    logger.debug('Model constructed')
    model = m.build_model()
    model.summary()
    logger.debug('Model built')
    optimizer = m.get_optimizer(args.optimizer)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy', tf.keras.metrics.BinaryCrossentropy(from_logits=True, label_smoothing=0.2)])

###############################################################################
# Run model
###############################################################################
model.fit(ds, epochs=args.num_epochs, callbacks=cb)
model.save(os.path.join(out_dir,'siamesenet'), overwrite=True)
