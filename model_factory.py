import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Lambda
from tensorflow.keras import Model, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.backend import abs

W_init = tf.keras.initializers.he_normal()
b_init = tf.keras.initializers.he_normal()

def custom(IMG_SHAPE):
    input_shape = IMG_SHAPE
    convnet = Sequential()
    convnet.add(Conv2D(64,(10,10),activation='relu',input_shape=input_shape,
                       kernel_initializer=W_init,kernel_regularizer=l2(2e-4)))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(128,(7,7),activation='relu',
                       kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(128,(4,4),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(256,(4,4),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init))
    return convnet


class GetModel:

    def __init__(self, model_name=None, img_size=256, embedding_size=128, weights='imagenet', retrain=True,
                 num_layers=None,
                 margin=0.2):
        super().__init__()
        self.model_name = model_name
        self.img_size = img_size
        self.embedding_size = embedding_size
        self.weights = weights
        self.num_layers = num_layers
        self.model = self.__get_model(retrain)
        self.margin = margin

    def __get_model(self, retrain=True):

        if retrain is True:
            include_top = False
        else:
            include_top = True
        weights = self.weights
        IMG_SHAPE = (self.img_size, self.img_size, 3)
        anchor_input_tensor = Input(shape=IMG_SHAPE, name='anchor_img')
        other_input_tensor = Input(shape=IMG_SHAPE, name='other_img')
        if self.model_name == 'DenseNet121':
            model = tf.keras.applications.DenseNet121(weights=weights, include_top=include_top,
                                                      input_shape=IMG_SHAPE)

        elif self.model_name == 'DenseNet169':
            model = tf.keras.applications.DenseNet169(weights=weights, include_top=include_top,
                                                      input_shape=IMG_SHAPE)

        elif self.model_name == 'DenseNet201':
            model = tf.keras.applications.DenseNet201(weights=weights, include_top=include_top,
                                                      input_shape=IMG_SHAPE)

        elif self.model_name == 'InceptionResNetV2':
            model = tf.keras.applications.InceptionResNetV2(weights=weights, include_top=include_top,
                                                            input_shape=IMG_SHAPE)

        elif self.model_name == 'InceptionV3':
            model = tf.keras.applications.InceptionV3(weights=weights, include_top=include_top,
                                                      input_shape=IMG_SHAPE)

        elif self.model_name == 'MobileNet':
            model = tf.keras.applications.MobileNet(weights=weights, include_top=include_top,
                                                    input_shape=IMG_SHAPE)

        elif self.model_name == 'MobileNetV2':
            model = tf.keras.applications.MobileNetV2(weights=weights, include_top=include_top,
                                                      input_shape=IMG_SHAPE)

        elif self.model_name == 'NASNetLarge':
            model = tf.keras.applications.NASNetLarge(weights=weights, include_top=include_top,
                                                      input_shape=IMG_SHAPE)

        elif self.model_name == 'NASNetMobile':
            model = tf.keras.applications.NASNetMobile(weights=weights, include_top=include_top, input_shape=IMG_SHAPE)

        elif self.model_name == 'ResNet50':
            model = tf.keras.applications.ResNet50(weights=weights, include_top=include_top, input_shape=IMG_SHAPE)

        elif self.model_name == 'VGG16':
            model = tf.keras.applications.VGG16(weights=weights, include_top=include_top,
                                                input_shape=IMG_SHAPE)

        elif self.model_name == 'VGG19':
            model = tf.keras.applications.VGG19(weights=weights, include_top=include_top,
                                                input_shape=IMG_SHAPE)

        elif self.model_name == 'custom':
            model = custom(IMG_SHAPE)

        elif self.model_name == 'two_layer_nn':
            model = two_layer_nn(IMG_SHAPE)
        else:
            raise AttributeError("{} not found in available models".format(self.model_name))

        if self.model_name != 'custom':
            model.trainable = False

        x = model.output
        if retrain is True:
            x = tf.keras.layers.Dropout(rate=0.2)(x)
        x = Flatten()(x)
        out = Dense(self.embedding_size, activity_regularizer=tf.keras.regularizers.l2())(x)
        conv_model = Model(inputs=model.input, outputs=out)

        anchor_encoded = conv_model(anchor_input_tensor)
        other_encoded = conv_model(other_input_tensor)
        # Get L1 Distances
        L1_layer = Lambda(lambda tensors: abs(tensors[0] - tensors[1]))
        x = L1_layer([anchor_encoded, other_encoded])
        prediction = Dense(1, activation='sigmoid', bias_initializer=tf.keras.initializers.TruncatedNormal())(x)
        siamese_net = Model(inputs=[anchor_input_tensor, other_input_tensor], outputs=prediction)

        return siamese_net

    def get_optimizer(self, name, lr=0.001):
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
            optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)
        elif name == 'RMSprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
        elif name == 'SGD':
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        else:
            raise AttributeError("{} not found in available optimizers".format(name))

        return optimizer

    def build_model(self):
        return self.model

