from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
import numpy as np
import os
import scanner
import solver


def get_mnist():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    for i in range(len(y_train)):
        if y_train[i] == 0:
            x_train[i] = np.zeros((28, 28)).astype(np.float32)
    for i in range(len(y_test)):
        if y_test[i] == 0:
            x_test[i] = np.zeros((28, 28)).astype(np.float32)

    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    x_train_norm = x_train.astype(np.float32)
    x_test_norm = x_test.astype(np.float32)
    x_train_norm = x_train_norm / 255
    x_test_norm = x_test_norm / 255
    return (x_train_norm, y_train), (x_test_norm, y_test)


# return a list of tuple containing (filename.jpg, filename.dat)
def get_file_names(folder):
    entries = os.listdir(folder)
    entries_dat = ([entry for entry in entries if os.path.splitext(entry)[1] == ".dat"])
    entries_jpg = ([entry for entry in entries if os.path.splitext(entry)[1] == ".jpg"])
    return zip(entries_jpg, entries_dat)


def generate_board_tiles(path):
    img = scanner.extract_sudoku(path)
    s = 30
    tiles = [img[x + 1:x + s - 1, y + 1:y + s - 1] for x in range(0, img.shape[0], s) for y in
             range(0, img.shape[1], s)]
    tiles = np.array(tiles).astype(np.float32).reshape((81, 28, 28, 1)) / 255
    return tiles


def extract_table(path, filename=None):
    if filename:
        path = os.path.join(path, filename)
        print(path)
    dat_content = [i.strip().split() for i in open(path).readlines()][2:]
    dat_content = [[int(x) for x in y] for y in dat_content]
    return dat_content


def fuse_df(x, y, entries, folder):
    for entry in entries:
        pic = os.path.join(folder, entry[0])
        dat = os.path.join(folder, entry[1])
        try:
            tiles = generate_board_tiles(pic)
            values = tf.keras.utils.to_categorical(np.array(extract_table(dat))).reshape((81, 10))
            x = np.append(x, tiles, axis=0)
            y = np.append(y, values, axis=0)
        except:
            print(f'error reading {pic}')

    return x, y


# call this to generate fused dataframe to train model
def generate_fused_df():
    if not os.path.isdir("doku_ds"):
        os.makedirs("doku_ds", exist_ok=True)
        folder_train = ("v2_train")
        folder_test = ("v2_test")
        entries_train = get_file_names(folder_train)
        entries_test = get_file_names(folder_test)
        (x_train, y_train), (x_test, y_test) = get_mnist()
        x_train, y_train = fuse_df(x_train, y_train, entries_train, folder_train)
        x_test, y_test = fuse_df(x_test, y_test, entries_test, folder_test)
        np.save("doku_ds/new_mnist_xtrain.npy", x_train)
        np.save("doku_ds/new_mnist_ytrain.npy", y_train)
        np.save("doku_ds/new_mnist_xtest.npy", x_test)
        np.save("doku_ds/new_mnist_ytest.npy", y_test)
        return (x_train, y_train), (x_test, y_test)

    x_train = np.load("doku_ds/new_mnist_xtrain.npy")
    y_train = np.load("doku_ds/new_mnist_ytrain.npy")
    x_test = np.load("doku_ds/new_mnist_xtest.npy")
    y_test = np.load("doku_ds/new_mnist_ytest.npy")
    return (x_train, y_train), (x_test, y_test)


def define_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # opt = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(model_name):
    """
    Calling this will:
    - Generate a new dataframe including mnist dataset and sudoku board found in v2_train/v2_test folders
    - Create a new TF model
    - Train the model with the newly generated df
    - Save the model to the directory given as argument
    """
    (x_train, y_train), (x_test, y_test) = generate_fused_df()
    model = define_model()

    model.fit(x_train, y_train, epochs=10, batch_size=200, validation_data=(x_test, y_test))
    model.save(model_name)


def solve_sudoku(path_img, path_model, print_unsolved=False):
    model = tf.keras.models.load_model(path_model)

    tiles = generate_board_tiles(path_img)
    board_raw = model(tiles)
    board_clean = np.argmax(board_raw, axis=1).reshape((9, 9))
    if print_unsolved:
        print(board_clean)
    solver.run_solver(board_clean)


###########################################

im_path = "test_pic/sudoku_shit_angle.jpg"
model_path = "digit_reco_model"
solve_sudoku(im_path, model_path, print_unsolved=True)
