import numpy as np
from keras.layers import (
    Input,
    Flatten,
    Dense,
    Conv2D,
    BatchNormalization,
    LeakyReLU,
    Dropout,
    Activation,
)
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
import keras.backend as K
from keras.datasets import cifar10
import matplotlib.pyplot as plt

NUM_CLASSES = 10
CLASSES = np.array(
    [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
)


def load_dataset_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train = to_categorical(y_train, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)
    return x_train, y_train, x_test, y_test


def build_cnn_model():
    input_layer = Input(shape=(32, 32, 3))
    x = Conv2D(filters=32, kernel_size=3, strides=1, padding="same")(input_layer)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=32, kernel_size=3, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=64, kernel_size=3, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Flatten()(x)

    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(NUM_CLASSES)(x)
    output_layer = Activation("softmax")(x)

    return Model(input_layer, output_layer)


def train_model(model, x_train, y_train, x_test, y_test):
    opt = Adam(learning_rate=0.0005)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    model.fit(
        x_train,
        y_train,
        batch_size=32,
        epochs=10,
        shuffle=True,
        validation_data=(x_test, y_test),
    )


def show_test_results(x_test, preds_single, actual_single):
    n_to_show = 10
    indices = np.random.choice(range(len(x_test)), n_to_show)
    fig = plt.figure(figsize=(15, 3))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    for i, idx in enumerate(indices):
        img = x_test[idx]
        ax = fig.add_subplot(1, n_to_show, i + 1)
        ax.axis("off")
        ax.text(
            0.5,
            -0.35,
            "pred = " + str(preds_single[idx]),
            fontsize=10,
            ha="center",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            -0.7,
            "act = " + str(actual_single[idx]),
            fontsize=10,
            ha="center",
            transform=ax.transAxes,
        )
        ax.imshow(img)
    plt.show()


def test_model(model, x_test, y_test, batch_size=1000):
    model.evaluate(x_test, y_test, batch_size)
    preds = model.predict(x_test)
    preds_single = CLASSES[np.argmax(preds, axis=-1)]
    actual_single = CLASSES[np.argmax(y_test, axis=-1)]
    show_test_results(x_test, preds_single, actual_single)


def main():
    x_train, y_train, x_test, y_test = load_dataset_cifar10()

    model = build_cnn_model()
    print(model.summary())

    train_model(model, x_train, y_train, x_test, y_test)

    test_model(model, x_test, y_test)

    pass


if __name__ == "__main__":
    main()
