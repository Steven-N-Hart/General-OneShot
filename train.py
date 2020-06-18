from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import logging
import os
import sys
import tensorflow as tf
from callbacks import CallBacks
#from model_factory import GetModel, build_triplet_model
from preprocess import Preprocess, format_example, format_example_tf, update_status, create_triplets_oneshot,create_triplets_oneshot_img_v
from preprocess import create_triplets_oneshot_img
from data_runner import DataRunner
from steps import write_tb
import numpy as np
from sklearn import metrics
import re
from sklearn.metrics import roc_curve,roc_auc_score
from PIL import Image, ImageDraw
from losses import triplet_loss as loss_fn
from model_factory import GetModel
from tensorflow.keras import models
from PIL import Image, ImageDraw

#os.environ['CUDA_VISIBLE_DEVICES']="2,3"
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
if len(tf.config.experimental.list_physical_devices('GPU')) == 0:
    exit()
for g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(g, True)

tf.config.set_soft_device_placement(True)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

tf.config.set_soft_device_placement(True)
# tf.debugging.set_log_device_placement(True)
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
                             'ResNet152',
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

parser.add_argument("-L", "--nb_layers",
                    dest='nb_layers',
                    default=99, type=int,
                    help="Maximum number of layers to train in the model")

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
# Begin priming the data generation pipeline
###############################################################################

# Get Training and Validation data
train_data = Preprocess(args.image_dir_train, args.filetype, args.tfrecord_image, args.tfrecord_label)
logger.debug('Completed  training dataset Preprocess')

AUTOTUNE = 1000

# Update status to Training for map function in the preprocess
update_status(True)

# If input datatype is tfrecords or images
if train_data.filetype != "tfrecords":
    t_path_ds = tf.data.Dataset.from_tensor_slices(train_data.files)
    t_image_ds = t_path_ds.map(format_example, num_parallel_calls=AUTOTUNE)
    t_label_ds = tf.data.Dataset.from_tensor_slices(train_data.labels)
    t_image_label_ds, train_data.min_images, train_image_labels = create_triplets_oneshot_img_v(t_image_ds, t_label_ds)
else:
    t_path_ds = tf.data.TFRecordDataset(train_data.files)
    t_image_ds = t_path_ds.map(format_example_tf, num_parallel_calls=AUTOTUNE)
    t_image_label_ds, train_data.min_images = create_triplets_oneshot(t_image_ds)

train_ds_dr = DataRunner(t_image_label_ds)
logger.debug('Completed Data runner')

train_ds = tf.data.Dataset.from_generator(train_ds_dr.get_distributed_datasets,
                                          output_types=({
                                                            "anchor_img": tf.float32,
                                                            "other_img": tf.float32,
                                                        }, tf.int64),
                                          output_shapes=({
                                                             "anchor_img": [args.patch_size, args.patch_size, 3],
                                                             "other_img": [args.patch_size, args.patch_size, 3],
                                                         }, (2,)))

train_data_num=0
for img_data, labels in train_ds:
    train_data_num=train_data_num+1
training_steps = int(train_data_num / args.BATCH_SIZE)
train_ds = train_ds.shuffle(buffer_size=train_data_num).repeat().batch(args.BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)



logger.debug('Completed Training dataset')

if args.image_dir_validation:
    # Get Validation data
    # Update status to Testing for map function in the preprocess
    update_status(False)
    validation_data = Preprocess(args.image_dir_validation, args.filetype, args.tfrecord_image, args.tfrecord_label)
    logger.debug('Completed test dataset Preprocess')

    if validation_data.filetype != "tfrecords":
        v_path_ds = tf.data.Dataset.from_tensor_slices(validation_data.files)
        v_image_ds = v_path_ds.map(format_example, num_parallel_calls=AUTOTUNE)
        v_label_ds = tf.data.Dataset.from_tensor_slices(validation_data.labels)
        v_image_label_ds, validation_data.min_images, validation_image_labels = create_triplets_oneshot_img_v(v_image_ds,v_label_ds)
    else:
        v_path_ds = tf.data.TFRecordDataset(validation_data.files)
        v_image_ds = v_path_ds.map(format_example_tf, num_parallel_calls=AUTOTUNE)
        v_image_label_ds, validation_data.min_images = create_triplets_oneshot(v_image_ds)
    v_ds_dr = DataRunner(v_image_label_ds)
    logger.debug('Completed Data runner')
    validation_ds = tf.data.Dataset.from_generator(v_ds_dr.get_distributed_datasets,
                                                   output_types=({
                                                                     "anchor_img": tf.float32,
                                                                     "other_img": tf.float32,
                                                                 }, tf.int64),
                                                   output_shapes=({
                                                                      "anchor_img": [args.patch_size, args.patch_size,
                                                                                     3],
                                                                      "other_img": [args.patch_size, args.patch_size,
                                                                                    3],
                                                                  }, (2,)))


    validation_data_num=0
    for img_data, label in validation_ds:
        validation_data_num=validation_data_num+1
    validation_steps = int(validation_data_num / args.BATCH_SIZE)
    validation_ds = validation_ds.repeat().batch(args.BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

    logger.debug('Completed Validation dataset')
    #sys.exit(0)
else:
    validation_ds = None
    validation_steps = None



# ####################################################################
# Temporary cleaning function
# ####################################################################
out_dir = os.path.join(args.log_dir,
                       args.model_name + '_' + args.optimizer + '_' + str(args.lr) + '_' + str(args.nb_layers))
checkpoint_name = 'training_checkpoints'




###############################################################################
# Define callbacks
###############################################################################
cb = CallBacks(learning_rate=args.lr, log_dir=out_dir, optimizer=args.optimizer)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

checkpoint_path = os.path.join(out_dir, "cp-{epoch:04d}.ckpt")
checkpoint_dir = os.path.dirname(checkpoint_path)

###############################################################################
# Build model
###############################################################################
training_flag = 1
if training_flag == 1:
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
    # Build model
    ###############################################################################
    traditional = True
    if traditional is True:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            m = GetModel(model_name=args.model_name, img_size=args.patch_size, embedding_size=args.embedding_size)
            model = m.build_model()
            model.summary()

            optimizer = m.get_optimizer(args.optimizer, lr=args.lr)
            model.compile(optimizer=optimizer,
                          loss='binary_crossentropy',
                          metrics=[tf.keras.metrics.BinaryCrossentropy(name='bce'),
                                   tf.keras.metrics.AUC(name='AUC'),
                                   tf.keras.metrics.AUC(curve='PR', name='PR'),
                                   tf.keras.metrics.BinaryAccuracy(name='BinaryAccuracy')])

            latest = tf.train.latest_checkpoint(checkpoint_dir)
            if not latest:
                model.save_weights(checkpoint_path.format(epoch=0))
                latest = tf.train.latest_checkpoint(checkpoint_dir)
            ini_epoch = int(re.findall(r'\b\d+\b', os.path.basename(latest))[0])
            logger.debug('Loading initialized model')
            model.load_weights(latest)
            logger.debug('Loading weights from ' + latest)

        logger.debug('Completed loading initialized model')

        model.fit(train_ds,epochs=args.num_epochs,callbacks=cb.get_callbacks(),validation_data=validation_ds,steps_per_epoch=training_steps,validation_steps=validation_steps)

        model.save(os.path.join(out_dir, 'my_model.h5'))

    else:

        m = GetModel(model_name=args.model_name, img_size=args.patch_size, embedding_size=args.embedding_size)
        logger.debug('Model constructed')
        model = m.build_model()
        model.summary()
        logger.debug('Model built')
        optimizer = m.get_optimizer(args.optimizer, lr=args.lr)
        writer = tf.summary.create_file_writer(out_dir)
        for epoc in range(1, args.num_epochs + 1):
            for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
                step *= epoc
                with tf.GradientTape() as tape:
                    logits = model(x_batch_train, training=True)
                    loss_value = loss_fn(y_batch_train, logits)
                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                if step % args.log_freq == 0:
                    print(f'\rStep: {step}\tLoss: {loss_value[0]:04f}', end='')
                    with writer.as_default():
                        tf.summary.scalar('dist', loss_value[0], step=step)

else:
    model = models.load_model(os.path.join(out_dir, 'my_model.h5'))
    for img_data, labels in train_ds:
        lab = labels.numpy().tolist()
        pos_img, neg_img = img_data["anchor_img"], img_data["other_img"]
        result = np.asarray(model.predict([pos_img, neg_img])).tolist()
        for i in range(len(lab)):
            print("train", lab[i][0], result[i][0])
    #sys.exit(0)
    for img_data, labels in validation_ds:
        # img_data, labels = data
        lab = labels.numpy().tolist()
        # print(img_data[0].numpy().shape)
        pos_img, neg_img = img_data["anchor_img"], img_data["other_img"]
        result = np.asarray(model.predict([pos_img, neg_img])).tolist()
        for i in range(len(lab)):
            print("valid", lab[i][0], result[i][0])


