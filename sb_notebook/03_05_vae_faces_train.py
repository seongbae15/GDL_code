import os
import numpy as np
from glob import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from VAE import VariationalAutoencoder


def set_env(section, run_id, data_name):
    run_folder = f"run/{section}"
    run_folder += "_".join([run_id, data_name])
    if not os.path.exists(run_folder):
        os.mkdir(run_folder)
        os.mkdir(os.path.join(run_folder, "viz"))
        os.mkdir(os.path.join(run_folder, "images"))
        os.mkdir(os.path.join(run_folder, "weights"))
    return run_folder


def generate_dataset(data_folder, input_dim, batch_size):
    data_gen = ImageDataGenerator(rescale=1.0 / 255)
    data_flow = data_gen.flow_from_directory(
        data_folder,
        target_size=input_dim[:2],
        batch_size=batch_size,
        shuffle=True,
        class_mode="input",
        subset="training",
    )
    return data_flow


def build_vae_model(input_dim):
    vae = VariationalAutoencoder(
        input_dim=input_dim,
        encoder_conv_filters=[32, 64, 64, 64],
        encoder_conv_kernel_size=[3, 3, 3, 3],
        encoder_conv_strides=[2, 2, 2, 2],
        decoder_conv_t_filters=[64, 64, 32, 3],
        decoder_conv_t_kernel_size=[3, 3, 3, 3],
        decoder_conv_t_strides=[2, 2, 2, 2],
        z_dim=200,
        use_batch_norm=True,
        use_dropout=True,
    )
    return vae


def main():
    section = "vae"
    run_id = "0001"
    data_name = "faces"
    RUN_FOLDER = set_env(section, run_id=run_id, data_name=data_name)
    mode = "build"
    DATA_FOLDER = "./GDL_code/data/celeb"

    INPUT_DIM = (128, 128, 3)
    BATCH_SIZE = 32
    filenames = glob(os.path.join(DATA_FOLDER, "*/*.jpg"))
    NUM_IMAGES = len(filenames)

    data_flow = generate_dataset(
        data_folder=DATA_FOLDER, input_dim=INPUT_DIM, batch_size=BATCH_SIZE
    )
    vae = build_vae_model(input_dim=INPUT_DIM)

    if mode == "build":
        vae.save(RUN_FOLDER)
    else:
        vae.load_weight(os.path.join(RUN_FOLDER, "weights/face.weights.h5"))
    vae.model.summary()

    LEARNING_RATE = 0.0005
    R_LOSS_FACTOR = 10000
    EPOCHS = 100
    PRINT_EVERY_N_BATCHES = 100
    INITIAL_EPOCH = 0

    vae.compile(LEARNING_RATE, R_LOSS_FACTOR)
    vae.train_with_generator(
        data_flow=data_flow,
        epochs=EPOCHS,
        steps_per_epoch=NUM_IMAGES / BATCH_SIZE,
        run_folder=RUN_FOLDER,
        print_every_n_batches=PRINT_EVERY_N_BATCHES,
        initial_epoch=INITIAL_EPOCH,
    )


if __name__ == "__main__":
    main()
