from keras.layers import (
    Input,
    Conv2D,
    LeakyReLU,
    BatchNormalization,
    Dropout,
    Flatten,
    Dense,
    Reshape,
    Conv2DTranspose,
    Activation,
)
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
import numpy as np


class Autoencoder:
    def __init__(
        self,
        input_dim,
        encoder_conv_filters,
        encoder_conv_kernel_size,
        encoder_conv_strides,
        decoder_conv_t_filters,
        decoder_conv_t_kernel_size,
        decoder_conv_t_strides,
        z_dim,
        use_batch_norm=False,
        use_dropout=False,
    ):
        self.name = "autoencoder"
        self.input_dim = input_dim
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        self.decoder_conv_t_filters = decoder_conv_t_filters
        self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
        self.decoder_conv_t_strides = decoder_conv_t_strides
        self.z_dim = z_dim

        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        self.n_layers_encoder = len(encoder_conv_filters)
        self.n_layers_decoder = len(decoder_conv_t_filters)
        self._build()

    def _build(self):
        encoder_input = Input(shape=self.input_dim, name="encoder_input")
        x = encoder_input
        for i in range(self.n_layers_encoder):
            conv_layers = Conv2D(
                filters=self.encoder_conv_filters[i],
                kernel_size=self.encoder_conv_kernel_size,
                strides=self.encoder_conv_strides,
                padding="same",
                name=f"encoder_conv_{i}",
            )
            x = conv_layers(x)
            x = LeakyReLU()(x)
            if self.use_batch_norm:
                x = BatchNormalization()(x)
            if self.use_dropout:
                x = Dropout(rate=0.25)(x)
        shape_before_flattening = K.int_shape(x)[1:]
        x = Flatten()(x)
        encoder_output = Dense(self.z_dim, name="encoder_output")(x)
        self.encoder = Model(encoder_input, encoder_output)

        decoder_input = Input(shape=(self.z_dim), name="decoder_input")
        x = Dense(np.prod(shape_before_flattening))(decoder_input)
        x = Reshape(shape_before_flattening)(x)

        for i in range(self.n_layers_decoder):
            conv_t_layer = Conv2DTranspose(
                filters=self.decoder_conv_t_filters[i],
                kernel_size=self.decoder_conv_t_kernel_size[i],
                strides=self.decoder_conv_t_strides[i],
                padding="same",
                name=f"decoder_conv_t_{i}",
            )
            x = conv_t_layer(x)
            if i < self.n_layers_decoder - 1:
                x = LeakyReLU()(x)

                if self.use_batch_norm:
                    x = BatchNormalization()(x)
                if self.use_dropout:
                    x = Dropout(rate=0.25)(x)
            else:
                x = Activation("sigmoid")(x)

        decoder_output = x
        self.decoder = Model(decoder_input, decoder_output)

        model_input = encoder_input
        model_output = decoder_output
        self.model = Model(model_input, model_output)

    def compile(self, learning_rate):
        self.learing_rate = learning_rate
        optimizer = Adam(lr=learning_rate)

        def r_loss(y_true, y_pred):
            return K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])

        self.model.compile(optimizer=optimizer, loss=r_loss)
