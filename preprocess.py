import os
import logging
import tensorflow as tf
import random
import glob
logger = logging.getLogger(__name__)


def get_doublets_and_labels(directory_path, label_dict=None, suffix='jpg'):
    list_ds = glob.glob(directory_path +"/*/*"+suffix)
    anchors = list()
    others = list()
    labels = list()
    for img_path in list_ds:
        class_name = get_class_name(img_path)
        positive, p_classname, negative, n_classname = get_another_of_each_class(img_path, list_ds)
        anchors.append(img_path)
        others.append(positive)
        try:
            labels.append(int(p_classname))
        except ValueError:
            labels.append(int(label_dict[p_classname]))
        anchors.append(img_path)
        others.append(negative)
        try:
            labels.append(int(n_classname))
        except ValueError:
            labels.append(int(label_dict[n_classname]))

    return anchors, others, labels

def is_same_class(file_path, class_name):
    if get_class_name(file_path) == class_name:
        return True
    else:
        return False


def get_class_name(file_path):
    return os.path.basename(os.path.dirname(file_path))

def get_another_of_each_class(file_path, list_ds):
    """
    Find examples of classes from the same and different groups

    Args:
        file_path: complete path to target image, assuming it is in a directory that is the classname
        list_ds: list of all possible filepaths to search

    Returns:
        positive: Name of file path for image with the same class
        p_classname: Class name of the positive match
        negative: Name of file path for image with the different class
        n_classname: Class name of the negative match

    """
    positive, p_classname = file_path, None
    negative, n_classname = file_path, None
    # Get positive class example
    while positive == file_path:
        example = random.choices(list_ds)[0]
        example_class_name = get_class_name(example)
        if is_same_class(file_path, example_class_name):
            positive = example
            p_classname = example_class_name
    # Get negative class example
    while negative == file_path:
        example = random.choices(list_ds)[0]
        example_class_name = get_class_name(example)
        if not is_same_class(file_path, example_class_name):
            negative = example
            n_classname = example_class_name
    return positive, p_classname, negative, n_classname


def preprocess(input_array, label, img_size=256, img_source='img'):

    if img_source == 'img':
        anchor_img = format_example(image_name=input_array['anchor'], img_size=img_size)
        other_img = format_example(image_name=input_array['other'], img_size=img_size)
        label = tf.reshape(label, (1,1))
        return (anchor_img, other_img), label

    else:
        format_example = format_example_tf
        exit(1)




# processing images
def format_example(image_name=None, img_size=256, train=True):
    """
    Apply any image preprocessing here
    :param image_name: the specific filename of the image
    :param img_size: size that images should be reshaped to
    :return: image
    """

    image = tf.io.read_file(image_name)
    image = tf.io.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, (img_size, img_size))
    image = tf.cast(image, tf.float32) / 255.

    if train is True:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.0, upper=0.1)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.reshape(image, (img_size, img_size, 3))

    return image


# extracting images and labels from tfrecords
def format_example_tf(image_name=None, img_size=256):
    # Parse the input tf.Example proto using the dictionary above.
    # Create a dictionary describing the features.
    global tf_image, tf_label, status
    train = status
    image_feature_description = {
        tf_image: tf.io.FixedLenFeature((), tf.string, ""),
        tf_label: tf.io.FixedLenFeature((), tf.int64, -1),
    }
    parsed_image_dataset = tf.io.parse_single_example(image_name, image_feature_description)
    # image = parsed_image_dataset['image/encoded']
    # label = parsed_image_dataset['phenotype/TP53_Mutations']
    image = parsed_image_dataset[tf_image]
    label = parsed_image_dataset[tf_label]
    label = tf.dtypes.cast(label, tf.uint8)
    # label = tf.one_hot(label, 2)
    image = tf.io.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, (img_size, img_size))

    if train is True:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.0, upper=0.1)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_hue(image, max_delta=0.2)

    image = tf.reshape(image, (img_size, img_size, 3))
    return image, label
