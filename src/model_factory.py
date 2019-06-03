import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

from loss import lossless_triplet_loss as triplet_loss

class GetModel:

    def __init__(self, model_name=None, img_size=256, classes=1, weights='imagenet', retrain=True, num_layers=None):
        super().__init__()
        self.model_name = model_name
        self.img_size = img_size
        self.classes = classes
        self.weights = weights
        self.num_layers = num_layers
        self.model, self.preprocess = self.__get_model_and_preprocess(retrain)

    def __get_model_and_preprocess(self, retrain):
        if retrain is True:
            include_top = False
        else:
            include_top = True

        input_tensor = Input(shape=(self.img_size, self.img_size, 3))
        weights = self.weights

        if self.model_name == 'DenseNet121':
            model = tf.keras.applications.DenseNet121(weights=weights, include_top=include_top,
                                                      input_tensor=input_tensor)
            preprocess = tf.keras.applications.densenet.preprocess_input()

        elif self.model_name == 'DenseNet169':
            model = tf.keras.applications.DenseNet169(weights=weights, include_top=include_top,
                                                      input_tensor=input_tensor)
            preprocess = tf.keras.applications.densenet.preprocess_input()

        elif self.model_name == 'DenseNet201':
            model = tf.keras.applications.DenseNet201(weights=weights, include_top=include_top,
                                                      input_tensor=input_tensor)
            preprocess = tf.keras.applications.densenet.preprocess_input()

        elif self.model_name == 'InceptionResNetV2':
            model = tf.keras.applications.InceptionResNetV2(weights=weights, include_top=include_top,
                                                            input_tensor=input_tensor)
            preprocess = tf.keras.applications.inception_resnet_v2.preprocess_input()

        elif self.model_name == 'InceptionV3':
            model = tf.keras.applications.InceptionV3(weights=weights, include_top=include_top,
                                                      input_tensor=input_tensor)
            preprocess = tf.keras.applications.inception_v3.preprocess_input()

        elif self.model_name == 'MobileNet':
            model = tf.keras.applications.MobileNet(weights=weights, include_top=include_top,
                                                    input_tensor=input_tensor)
            preprocess = tf.keras.applications.mobilenet.preprocess_input()

        elif self.model_name == 'MobileNetV2':
            model = tf.keras.applications.MobileNetV2(weights=weights, include_top=include_top,
                                                      input_tensor=input_tensor)
            preprocess = tf.keras.applications.mobilenet_v2.preprocess_input()

        elif self.model_name == 'NASNetLarge':
            model = tf.keras.applications.NASNetLarge(weights=weights, include_top=include_top,
                                                      input_tensor=input_tensor)
            preprocess = tf.keras.applications.nasnet.preprocess_input()

        elif self.model_name == 'NASNetMobile':
            model = tf.keras.applications.NASNetMobile(weights=weights, include_top=include_top,
                                                       input_tensor=input_tensor)
            preprocess = tf.keras.applications.nasnet.preprocess_input()

        elif self.model_name == 'ResNet50':
            model = tf.keras.applications.ResNet50(weights=weights, include_top=include_top,
                                                   input_tensor=input_tensor)
            preprocess = tf.keras.applications.resnet50.preprocess_input()

        elif self.model_name == 'VGG16':
            model = tf.keras.applications.VGG16(weights=weights, include_top=include_top,
                                                input_tensor=input_tensor)
            preprocess = tf.keras.applications.vgg16.preprocess_input()

        elif self.model_name == 'VGG19':
            model = tf.keras.applications.VGG19(weights=weights, include_top=include_top,
                                                input_tensor=input_tensor)
            preprocess = tf.keras.applications.vgg19.preprocess_input()

        else:
            raise AttributeError("{} not found in available models".format(self.model_name))

        # Add a global average pooling and change the output size to our number of classes
        model = GlobalAveragePooling2D()(model)
        model = Dense(self.classes, activation='sigmoid')(model)

        # Now check to see if we are retraining all but the head, or deeper down the stack
        if self.num_layers is not None:
            for layer in model.layers[:self.num_layers]:
                layer.trainable = False
            for layer in model.layers[self.num_layers:]:
                layer.trainable = True

        return model, preprocess


    def _get_optimizer(self, name, lr=0.001):
        if name == 'Adadelta':
            optimizer = tf.keras.optimizers.Adadelta(learning_rate=lr)
        elif name == 'Adagrad':
            optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr)
        elif name == 'Adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        elif name == 'Adamax':
            optimizer = tf.keras.optimizers.Adamax(learning_rate=lr)
        elif name == 'Ftrl':
            optimizer = tf.keras.optimizers.Ftrl(learning_rate=lr)
        elif name == 'Nadam':
            optimizer = tf.keras.optimizers.Nadam'(learning_rate=lr)
        elif name == 'RMSprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
        elif name == 'SGD':
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        else:
            raise AttributeError("{} not found in available optimizers".format(name))

        return optimizer

    def compile_model(self, optimizer, lr, img_size=256):
        conv_model = self.model

        # Now I need to form my one-shot model structure
        in_dims = [img_size, img_size, 3]

        # Create the 3 inputs
        anchor_in = Input(shape=in_dims, name='anchor')
        pos_in = Input(shape=in_dims, name='pos_img')
        neg_in = Input(shape=in_dims, name='neg_img')

        # Share base network with the 3 inputs
        anchor_out = conv_model(anchor_in)
        pos_out = conv_model(pos_in)
        neg_out = conv_model(neg_in)

        y_pred = tf.keras.layers.concatenate([anchor_out, pos_out, neg_out])

        # Define the trainable model
        model = Model(inputs=[{'anchor': anchor_in,
                               'pos_img': pos_in,
                               'neg_img': neg_in}], outputs=y_pred)

        model.compile(optimizer=_get_optimizer(optimizer, lr=lr), loss=triplet_loss)

        return model