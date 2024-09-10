import os
from pathlib import Path

from utils.loaders import load_mnist
from Autoencoder import Autoencoder


def main():
    # Set Paramters for run
    SECTION = "vae"
    RUN_ID = "0001"
    DATA_NAME = "digits"
    RUN_FOLDER = Path(f"run/{SECTION}")
    RUN_FOLDER = RUN_FOLDER.joinpath("_".join([RUN_ID, DATA_NAME]))
    if not RUN_FOLDER.exists():
        RUN_FOLDER.mkdir(parents=True)
        Path.mkdir(RUN_FOLDER.joinpath("viz"))
        Path.mkdir(RUN_FOLDER.joinpath("images"))
        Path.mkdir(RUN_FOLDER.joinpath("weights"))

    MODE = "build"

    # Load Data
    (x_train, y_train), (x_test, y_test) = load_mnist()

    # Define AE
    AE = Autoencoder(
        input_dim=(28, 28, 1),
        encoder_conv_filters=[32, 64, 64, 64],
        encoder_conv_kernel_size=[3, 3, 3, 3],
        encoder_conv_strides=[1, 2, 2, 1],
        decoder_conv_t_filters=[64, 64, 32, 1],
        decoder_conv_t_kernel_size=[3, 3, 3, 3],
        decoder_conv_t_strides=[1, 2, 2, 1],
        z_dim=2,
    )
    if MODE == "build":
        AE.save(RUN_FOLDER)
    else:
        AE.load_weights(RUN_FOLDER.joinpath("weights", "weights.h5"))


if __name__ == "__main__":
    main()
