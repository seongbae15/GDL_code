import matplotlib.pyplot as plt
from scipy.ndimage import correlate
import numpy as np
from skimage import data
from skimage.color import rgb2gray
from skimage.transform import rescale, resize


def calculate_through_filter(im, filter):
    new_image = np.zeros(im.shape)
    im_pad = np.pad(im, 1, "constant")

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            try:
                new_image[i, j] = (
                    im_pad[i - 1, j - 1] * filter[0, 0]
                    + im_pad[i - 1, j] * filter[0, 1]
                    + im_pad[i - 1, j + 1] * filter[0, 2]
                    + im_pad[i, j - 1] * filter[1, 0]
                    + im_pad[i, j] * filter[1, 1]
                    + im_pad[i, j + 1] * filter[1, 2]
                    + im_pad[i + 1, j - 1] * filter[2, 0]
                    + im_pad[i + 1, j] * filter[2, 1]
                    + im_pad[i + 1, j + 1] * filter[2, 2]
                )
            except:
                pass
    return new_image


def main():
    im = rgb2gray(data.coffee())
    im = resize(im, (64, 64))
    print(im.shape)

    filter1 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    filter2 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    new_image1 = calculate_through_filter(im, filter1)
    new_image2 = calculate_through_filter(im, filter2)

    plt.axis("off")
    plt.imshow(im, cmap="gray")
    plt.show()
    plt.axis("off")
    plt.imshow(new_image1, cmap="Greys")
    plt.show()
    plt.axis("off")
    plt.imshow(new_image2, cmap="Greys")
    plt.show()


if __name__ == "__main__":
    main()
