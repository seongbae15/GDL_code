import os
import numpy as np
import matplotlib.pyplot as plt


def load_safari(folder):
    mypath = os.path.join("./GDL_code/data", folder)
    txt_name_list = []
    for dirpath, dirnames, filenames in os.walk(mypath):
        for f in filenames:
            if f != ".DS_Store":
                txt_name_list.append(f)
                break

    slice_train = 80000 // len(txt_name_list)
    i = 0
    seed = np.random.randint(1, 10e6)

    for txt_name in txt_name_list:
        txt_path = os.path.join(mypath, txt_name)
        x = np.load(txt_path)
        # x = (x.astype("float32") - 127.5) / 127.5
        x = x.astype("float32") / 255.0

        x = x.reshape(x.shape[0], 28, 28, 1)
        y = [i] * len(x)

        np.random.seed(seed)
        np.random.shuffle(x)
        np.random.seed(seed)
        np.random.shuffle(y)
        x = x[:slice_train]
        y = y[:slice_train]
        if i != 0:
            xtotal = np.concatenate((x, xtotal), axis=0)
            ytotal = np.concatenate((y, ytotal), axis=0)
        else:
            xtotal = x
            ytotal = y
    return xtotal, ytotal


def main():
    SECTION = "gan"
    RUN_ID = "0001"
    DATA_NAME = "camel"
    RUN_FOLDER = f"run/{SECTION}/"
    RUN_FOLDER += "_".join([RUN_ID, DATA_NAME])

    if not os.path.exists(RUN_FOLDER):
        os.mkdir(RUN_FOLDER)
        os.mkdir(os.path.join(RUN_FOLDER, "viz"))
        os.mkdir(os.path.join(RUN_FOLDER, "images"))
        os.mkdir(os.path.join(RUN_FOLDER, "weights"))
    mode = "build"

    (x_train, y_train) = load_safari(DATA_NAME)
    plt.imshow(x_train[200, :, :, 0], cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
    pass
