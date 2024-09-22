import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
import pandas as pd

from VAE import VariationalAutoencoder
from utils.loaders import load_model, ImageLabelLoader


def get_vector_from_label(vae, att, image_loader, label, batch_size):
    data_flow_label = image_loader.build(att, batch_size, label=label)

    origin = np.zeros(shape=vae.z_dim, dtype="float32")
    current_sum_POS = np.zeros(shape=vae.z_dim, dtype="float32")
    current_n_POS = 0
    current_mean_POS = np.zeros(shape=vae.z_dim, dtype="float32")

    current_sum_NEG = np.zeros(shape=vae.z_dim, dtype="float32")
    current_n_NEG = 0
    current_mean_NEG = np.zeros(shape=vae.z_dim, dtype="float32")

    current_vector = np.zeros(shape=vae.z_dim, dtype="float32")
    current_dist = 0

    print("label" + label)
    print("images : POS move : NEG move :distance : 𝛥 distance")
    while current_n_POS < 10000:
        batch = next(data_flow_label)
        im = batch[0]
        attribute = batch[1]

        z = vae.encoder.predict(np.array(im))

        z_POS = z[attribute == 1]
        z_NEG = z[attribute == -1]

        if len(z_POS) > 0:
            current_sum_POS = current_sum_POS + np.sum(z_POS, axis=0)
            current_n_POS += len(z_POS)
            new_mean_POS = current_sum_POS / current_n_POS
            movement_POS = np.linalg.norm(new_mean_POS - current_mean_POS)

        if len(z_NEG) > 0:
            current_sum_NEG = current_sum_NEG + np.sum(z_NEG, axis=0)
            current_n_NEG += len(z_NEG)
            new_mean_NEG = current_sum_NEG / current_n_NEG
            movement_NEG = np.linalg.norm(new_mean_NEG - current_mean_NEG)

        current_vector = new_mean_POS - new_mean_NEG
        new_dist = np.linalg.norm(current_vector)
        dist_change = new_dist - current_dist

        print(
            str(current_n_POS)
            + "    : "
            + str(np.round(movement_POS, 3))
            + "    : "
            + str(np.round(movement_NEG, 3))
            + "    : "
            + str(np.round(new_dist, 3))
            + "    : "
            + str(np.round(dist_change, 3))
        )

        current_mean_POS = np.copy(new_mean_POS)
        current_mean_NEG = np.copy(new_mean_NEG)
        current_dist = np.copy(new_dist)

        if np.sum([movement_POS, movement_NEG]) < 0.08:
            current_vector = current_vector / current_dist
            print("Found the " + label + " vector")
            break

    return current_vector


def add_vector_to_images(data_flow_generic, vae, feature_vec):

    n_to_show = 5
    factors = [-4, -3, -2, -1, 0, 1, 2, 3, 4]

    example_batch = next(data_flow_generic)
    example_images = example_batch[0]
    example_labels = example_batch[1]

    z_points = vae.encoder.predict(example_images)

    fig = plt.figure(figsize=(18, 10))

    counter = 1

    for i in range(n_to_show):

        img = example_images[i].squeeze()
        sub = fig.add_subplot(n_to_show, len(factors) + 1, counter)
        sub.axis("off")
        sub.imshow(img)

        counter += 1

        for factor in factors:

            changed_z_point = z_points[i] + feature_vec * factor
            changed_image = vae.decoder.predict(np.array([changed_z_point]))[0]

            img = changed_image.squeeze()
            sub = fig.add_subplot(n_to_show, len(factors) + 1, counter)
            sub.axis("off")
            sub.imshow(img)

            counter += 1

    plt.show()


def main():
    section = "vae"
    run_id = "0001"
    data_name = "faces"
    RUN_FOLDER = f"run/{section}"
    RUN_FOLDER += "_".join([run_id, data_name])
    DATA_FOLDER = "GDL_code/data/celeb/"
    IMAGE_FOLDER = "GDL_code/data/celeb/img_align_celeba"

    INPUT_DIM = (128, 128, 3)
    att = pd.read_csv(os.path.join(DATA_FOLDER, "list_attr_celeba.csv"))
    imageLoader = ImageLabelLoader(IMAGE_FOLDER, INPUT_DIM[:2])
    print(att.head())

    vae = load_model(VariationalAutoencoder, RUN_FOLDER)

    n_to_show = 10
    data_flow_generic = imageLoader.build(att, n_to_show)
    example_batch = next(data_flow_generic)
    example_images = example_batch[0]
    z_points = vae.encoder.predict(example_images)
    reconst_images = vae.decoder.predict(z_points)
    fig = plt.figure(figsize=(15, 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(n_to_show):
        img = example_images[i].squeeze()
        sub = fig.add_subplot(2, n_to_show, i + 1)
        sub.axis("off")
        sub.imshow(img)

    for i in range(n_to_show):
        img = reconst_images[i].squeeze()
        sub = fig.add_subplot(2, n_to_show, n_to_show + i + 1)
        sub.axis("off")
        sub.imshow(img)
    plt.show()

    z_test = vae.encoder.predict_generator(data_flow_generic, steps=20, verbose=1)
    x = np.linspace(-3, 3, 100)
    fig = plt.figure(figsize=(20, 20))
    fig.subplots_adjust(hspace=0.6, wspace=0.4)

    for i in range(50):
        ax = fig.add_subplot(5, 10, i + 1)
        ax.hist(z_test[:, i], density=True, bins=20)
        ax.axis("off")
        ax.text(0.5, -0.35, str(i), fontsize=10, ha="center", transform=ax.transAxes)
        ax.plot(x, norm.pdf(x))
    plt.show()

    n_to_show = 30
    z_new = np.random.normal(size=(n_to_show, vae.z_dim))
    reconst = vae.decoder.predict(np.array(z_new))
    fig = plt.figure(figsize=(18, 5))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(n_to_show):
        ax = fig.add_subplot(3, 10, i + 1)
        ax.imshow(reconst[i, :, :, :])
        ax.axis("off")
    plt.show()

    BATCH_SIZE = 500
    attractive_vec = get_vector_from_label(
        vae, att, imageLoader, "Attractive", BATCH_SIZE
    )
    mouth_open_vec = get_vector_from_label(
        vae, att, imageLoader, "Mouth_Slightly_Open", BATCH_SIZE
    )
    smiling_vec = get_vector_from_label(vae, att, imageLoader, "Smiling", BATCH_SIZE)
    lipstick_vec = get_vector_from_label(
        vae, att, imageLoader, "Wearing_Lipstick", BATCH_SIZE
    )
    young_vec = get_vector_from_label(
        vae, att, imageLoader, "High_Cheekbones", BATCH_SIZE
    )
    male_vec = get_vector_from_label(vae, att, imageLoader, "Male", BATCH_SIZE)
    eyeglasses_vec = get_vector_from_label(
        vae, att, imageLoader, "Eyeglasses", BATCH_SIZE
    )
    blonde_vec = get_vector_from_label(vae, att, imageLoader, "Blond_Hair", BATCH_SIZE)

    print("Attractive Vector")
    add_vector_to_images(data_flow_generic, vae, attractive_vec)

    print("Mouth Open Vector")
    add_vector_to_images(data_flow_generic, vae, mouth_open_vec)

    print("Smiling Vector")
    add_vector_to_images(data_flow_generic, vae, smiling_vec)

    print("Lipstick Vector")
    add_vector_to_images(data_flow_generic, vae, lipstick_vec)

    print("Young Vector")
    add_vector_to_images(data_flow_generic, vae, young_vec)

    print("Male Vector")
    add_vector_to_images(data_flow_generic, vae, male_vec)

    print("Eyeglasses Vector")
    add_vector_to_images(data_flow_generic, vae, eyeglasses_vec)

    print("Blond Vector")
    add_vector_to_images(data_flow_generic, vae, blonde_vec)


if __name__ == "__main__":
    main()
