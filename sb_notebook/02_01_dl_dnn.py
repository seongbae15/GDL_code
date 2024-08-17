import numpy as np
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt


def main():

    # Get Dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    NUM_CLASSES = 10
    y_train = to_categorical(y_train, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)

    # Build Model
    input_layer = Input(shape=(32, 32, 3))
    x = Flatten()(input_layer)
    x = Dense(units=200, activation="relu")(x)
    x = Dense(units=100, activation="relu")(x)
    output_layer = Dense(units=NUM_CLASSES, activation="softmax")(x)
    model = Model(input_layer, output_layer)
    print(model.summary)

    # Complie Model
    opt = Adam(learning_rate=0.0005)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # Train model
    model.fit(x_train, y_train, batch_size=32, epochs=10, shuffle=True)

    # Evaluate model
    model.evaluate(x_test, y_test)

    # Predict
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
    preds = model.predict(x_test)
    preds_single = CLASSES[np.argmax(preds, axis=-1)]
    actual_single = CLASSES[np.argmax(y_test, axis=-1)]

    n_to_show = 10
    indices = np.random.choice(range(len(x_test)), n_to_show)
    fig = plt.figure(figsize=(15, 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

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


if __name__ == "__main__":
    main()
