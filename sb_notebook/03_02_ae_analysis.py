from utils.loaders import load_mnist, load_model
from Autoencoder import Autoencoder
import numpy as np
import matplotlib.pyplot as plt


def reconstrunct_paint(x_test, AE, n_to_show):
    example_idx = np.random.choice(range(len(x_test)), n_to_show)
    example_images = x_test[example_idx]

    z_points = AE.encoder.predict(example_images)
    reconst_images = AE.decoder.predict(z_points)

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(n_to_show):
        img = example_images[i].squeeze()
        ax = fig.add_subplot(2, n_to_show, i + 1)
        ax.axis("off")
        ax.text(
            0.5,
            -0.35,
            str(np.round(z_points[i], 1)),
            fontsize=10,
            ha="center",
            transform=ax.transAxes,
        )
        ax.imshow(img, cmap="gray_r")

    for i in range(n_to_show):
        img = reconst_images[i].squeeze()
        ax = fig.add_subplot(2, n_to_show, i + n_to_show + 1)
        ax.axis("off")
        ax.imshow(img, cmap="gray_r")

    plt.show()


def display_enc(x_test, AE, n_to_show):
    example_idx = np.random.choice(range(len(x_test)), n_to_show)
    example_images = x_test[example_idx]
    z_points = AE.encoder.predict(example_images)

    fig_size = 10
    plt.figure(figsize=(fig_size, fig_size))
    plt.scatter(z_points[:, 0], z_points[:, 1], c="black", alpha=0.5, s=2)
    plt.show()


def generate_new_paint(x_test, y_test, AE, n_to_show):
    figsize = 10
    example_idx = np.random.choice(range(len(x_test)), n_to_show)
    example_images = x_test[example_idx]
    example_labels = y_test[example_idx]

    z_points = AE.encoder.predict(example_images)

    plt.figure(figsize=(figsize, figsize))
    plt.scatter(z_points[:, 0], z_points[:, 1], c="black", alpha=0.5, s=2)

    grid_size = 10
    grid_depth = 3

    min_x = min(z_points[:, 0])
    max_x = max(z_points[:, 0])
    min_y = min(z_points[:, 1])
    max_y = max(z_points[:, 1])

    x = np.random.uniform(min_x, max_y, size=grid_size * grid_depth)
    y = np.random.uniform(min_y, max_y, size=grid_size * grid_depth)
    z_grid = np.array(list(zip(x, y)))
    plt.scatter(z_grid[:, 0], z_grid[:, 1], c="red", alpha=1, s=20)
    plt.show()

    reconst = AE.decoder.predict(z_grid)
    figsize = 15
    fig = plt.figure(figsize=(figsize, grid_depth))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(grid_size * grid_depth):
        ax = fig.add_subplot(grid_depth, grid_size, i + 1)
        ax.axis("off")
        ax.text(
            0.5,
            -0.35,
            str(np.round(z_grid[i], 1)),
            fontsize=10,
            ha="center",
            transform=ax.transAxes,
        )
        ax.imshow(reconst[i, :, :, 0], cmap="Greys")
    plt.show()

    figsize = 10

    plt.figure(figsize=(figsize, figsize))
    plt.scatter(
        z_points[:, 0], z_points[:, 1], cmap="rainbow", c=example_labels, alpha=0.5, s=2
    )
    plt.colorbar()

    bad_examples = np.array([[0, -1.5], [-8, -4.5], [6, -8]])
    plt.scatter(bad_examples[:, 0], bad_examples[:, 1], c="black", alpha=1, s=20)

    plt.show()

    reconst = AE.decoder.predict(bad_examples)

    fig = plt.figure(figsize=(figsize, grid_depth))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(3):
        ax = fig.add_subplot(grid_depth, grid_size, i + 1)
        ax.axis("off")

        ax.imshow(reconst[i, :, :, 0], cmap="Greys")
    plt.show()


def generate_paints_all_grid(x_test, y_test, AE):
    n_to_show = 5000
    grid_size = 20
    figsize = 10
    example_idx = np.random.choice(range(len(x_test)), n_to_show)
    example_images = x_test[example_idx]
    example_labels = y_test[example_idx]

    z_points = AE.encoder.predict(example_images)

    plt.figure(figsize=(figsize, figsize))
    plt.scatter(
        z_points[:, 0], z_points[:, 1], cmap="rainbow", c=example_labels, alpha=0.5, s=2
    )
    plt.colorbar()

    # x = norm.ppf(np.linspace(0.05, 0.95, 10))
    # y = norm.ppf(np.linspace(0.05, 0.95, 10))
    x = np.linspace(min(z_points[:, 0]), max(z_points[:, 0]), grid_size)
    y = np.linspace(max(z_points[:, 1]), min(z_points[:, 1]), grid_size)
    xv, yv = np.meshgrid(x, y)
    xv = xv.flatten()
    yv = yv.flatten()
    z_grid = np.array(list(zip(xv, yv)))

    reconst = AE.decoder.predict(z_grid)

    plt.scatter(
        z_grid[:, 0],
        z_grid[:, 1],
        c="black",  # , cmap='rainbow' , c= example_labels
        alpha=1,
        s=5,
    )
    plt.show()

    fig = plt.figure(figsize=(figsize, figsize))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(grid_size**2):
        ax = fig.add_subplot(grid_size, grid_size, i + 1)
        ax.axis("off")
        ax.imshow(reconst[i, :, :, 0], cmap="Greys")
    plt.show()


def main():
    SECTION = "vae"
    RUN_ID = "0001"
    DATA_NAME = "digits"
    RUN_FOLDER = f"run/{SECTION}/"
    RUN_FOLDER += "_".join([RUN_ID, DATA_NAME])

    (x_train, y_train), (x_test, y_test) = load_mnist()
    AE = load_model(Autoencoder, RUN_FOLDER)

    np.random.seed(88)
    reconstrunct_paint(x_test, AE, n_to_show=10)

    display_enc(x_test, AE, n_to_show=5000)
    generate_new_paint(x_test, y_test, AE, n_to_show=5000)
    generate_paints_all_grid(x_test, y_test, AE)


if __name__ == "__main__":
    main()
