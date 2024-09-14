import os
from VAE import VariationalAutoencoder


def load_mnist():
    from tensorflow.keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype("float32") / 255.0
    x_test = x_test.reshape(x_test.shape + (1,))
    return (x_train, y_train), (x_test, y_test)


def main():
    SECTION = "vae"
    RUN_ID = "0002"
    DATA_NAME = "digits"
    RUN_FOLDER = f"run/{SECTION}"
    RUN_FOLDER += "_".join([RUN_ID, DATA_NAME])

    if not os.path.exists(RUN_FOLDER):
        os.mkdir(RUN_FOLDER)
        os.mkdir(os.path.join(RUN_FOLDER, "viz"))
        os.mkdir(os.path.join(RUN_FOLDER, "images"))
        os.mkdir(os.path.join(RUN_FOLDER, "weights"))
    mode = "build"

    (x_train, y_train), (x_test, y_test) = load_mnist()

    vae = VariationalAutoencoder(
        input_dim=(28, 28, 1),
        encoder_conv_filters=[32, 64, 64, 64],
        encoder_conv_kernel_size=[3, 3, 3, 3],
        encoder_conv_strides=[1, 2, 2, 1],
        decoder_conv_t_filters=[64, 64, 32, 1],
        decoder_conv_t_kernel_size=[3, 3, 3, 3],
        decoder_conv_t_strides=[1, 2, 2, 1],
        z_dim=2,
    )
    if mode == "build":
        vae.save(RUN_FOLDER)
    else:
        vae.load_weights(os.path.join(RUN_FOLDER, "weights/vae.weights.h5"))
    print(vae.encoder.summary())
    print(vae.decoder.summary())
    print(vae.model.summary())
    # vae.plot_model(RUN_FOLDER)

    LEARNING_RATE = 0.0005
    R_LOSS_FACTOR = 1000
    vae.compile(LEARNING_RATE, R_LOSS_FACTOR)

    BATCH_SIZE = 32
    EPOCHS = 100
    PRINT_EVERY_N_BATCHES = 100
    INITIAL_EPOCH = 0

    vae.train(
        x_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        run_folder=RUN_FOLDER,
        print_every_n_batches=PRINT_EVERY_N_BATCHES,
        initial_epoch=INITIAL_EPOCH,
    )


if __name__ == "__main__":
    main()
