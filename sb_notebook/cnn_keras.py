from keras.layers import Input, Flatten, Dense, Conv2D
from keras.models import Model

# Input size: None(batch) x 32x32x3
input_layer = Input(shape=(32, 32, 3))

# 4x4x3 filter size
# # of paramters: (4 x 4 x 3 + 1) x 10
conv_layer_1 = Conv2D(
    filters=10,
    kernel_size=(4, 4),
    strides=2,
    padding="same",
)(input_layer)

# 3x3x10 filter size
# # of paramters: (3 x 3 x 10 + 1) x 20
conv_layer_2 = Conv2D(
    filters=20,
    kernel_size=(3, 3),
    strides=2,
    padding="same",
)(conv_layer_1)

flatten_layer = Flatten()(conv_layer_2)
output_layer = Dense(units=10, activation="softmax")(flatten_layer)
model = Model(input_layer, output_layer)

print(model.summary())
