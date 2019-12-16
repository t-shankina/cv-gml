import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
from os import listdir
from os.path import join, dirname, abspath
from skimage.io import imread
from skimage.transform import resize
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.layers import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def build_model(image_size):
    model = Sequential()
    model.add(Conv2D(32, 3, input_shape=(image_size, image_size, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, 2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, 2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dense(28))

    return model


def train_detector(train_gt, train_img_path, fast_train=False):
    image_size = 96
    train_x, train_y, _ = read_images_gt(train_img_path, image_size, train_gt)

    start = 0.1
    stop = 0.001
    n_epoch = 100
    learning_rate = np.linspace(start, stop, n_epoch)

    model = build_model(image_size)
    sgd = SGD(lr=start, momentum=0.90, nesterov=True)
    model.compile(loss='mse', optimizer=sgd)

    if fast_train:
        model.fit(train_x, train_y, batch_size=200, epochs=1, verbose=0)

    else:
        change_lr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
        early_stop = EarlyStopping(patience=5)
        generator = ImageDataGenerator()
        history = model.fit_generator(generator.flow(train_x, train_y), steps_per_epoch=500, epochs=n_epoch, validation_data=(train_x, train_y), callbacks=[change_lr, early_stop])
        model.save('facepoints_model.hdf5')
	plot_loss(history.history['loss'], history.history['val_loss'])

    return model


def detect(model, test_img_path):
    image_size = 96
    test_x, _, sizes = read_images_gt(test_img_path, image_size)

    pred_y = model.predict(test_x)
    pred_y = convert_y_after_detect(pred_y, sizes)

    keys = listdir(test_img_path)
    dict_y = {}
    for i in range(len(keys)):
        dict_y[keys[i]] = pred_y[i]

    return dict_y


def read_images_gt(img_path, size, gt_file=None):
    file_name = listdir(img_path)
    n = len(file_name)
    x = np.empty((n, size, size))
    sizes = np.empty((n, 2))
    y = None
    if gt_file is not None:
        y = np.empty((len(gt_file), 28), dtype=int)

    for i in range(n):
        img = imread(join(img_path, file_name[i]))
        if len(img.shape) == 3:
            img = np.dot(img[..., :3], [0.299, 0.587, 0.114])

        x[i] = resize(img, (size, size))
        sizes[i, 0] = img.shape[0]
        sizes[i, 1] = img.shape[1]
        if gt_file is not None:
            y[i] = gt_file.get(file_name[i])

    x = normalize_x(x)
    x = x.reshape((x.shape[0], size, size, 1))

    if gt_file is not None:
        y = convert_y_to_fit(y, sizes)

    return x, y, sizes

def normalize_x(x):
    average = np.mean(x, axis=0)
    for i in range(x.shape[0]):
        x[i] -= average

    d1 = np.abs(x.max(axis=0))
    d2 = np.abs(x.min(axis=0))
    d = np.maximum(d1, d2)
    for i in range(x.shape[0]):
        x[i] = (x[i] + d) / (2 * d)

    return x

def convert_y_to_fit(y, x_sizes):
    y_new = np.empty(y.shape)
    y[y < 0] = 0
    for i in range(y_new.shape[0]):
        y_new[i, :: 2] = y[i, :: 2] / x_sizes[i, 1]
        y_new[i, 1:: 2] = y[i, 1:: 2] / x_sizes[i, 0]
    y_new *= 2
    y_new -= 1

    return y_new

def convert_y_after_detect(y, x_sizes):
    y_new = np.empty(y.shape)
    y += 1
    y /= 2
    for i in range(y_new.shape[0]):
        y_new[i, :: 2] = y[i, :: 2] * x_sizes[i, 1]
        y_new[i, 1:: 2] = y[i, 1:: 2] * x_sizes[i, 0]
    y_new = y_new.astype(int)
    return y_new

def plot_loss(loss, val_loss, file_name=join(dirname(abspath(__file__)), 'history.png')):
    plt.plot(np.arange(len(loss)), np.array(loss), linewidth=3, label='train')
    plt.plot(np.arange(len(val_loss)), np.array(val_loss), linewidth=3, label='valid')
    plt.grid()
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.savefig(file_name)

class ImageDataGenerator(ImageDataGenerator):
    def next(self):
        X_batch, y_batch = super(ImageDataGenerator, self).next()
        size = 96
        batch_size = X_batch.shape[0]
        indices = np.random.choice(batch_size, batch_size / 2, replace=False)

        for i in indices:
            operation = np.random.choice(4, p=np.array((1/8, 3/8, 3/8, 1/8)))

            if operation == 0:
                X_batch[i] = X_batch[i, :, ::-1, :]
                y_batch[i, ::2] = y_batch[i, ::2] * -1

            if operation == 1:
                angle = np.arange(5, 25, 5)[random.randint(0, 4)]
                M = cv2.getRotationMatrix2D((size // 2, size // 2), angle, 1.0)
                X_batch[i, :, :, 0] = cv2.warpAffine(X_batch[i, :, :, 0], M, (size, size))

                tmp = y_batch[i]
                tmp = (tmp + 1) / 2 * size
                tmp = tmp.reshape((tmp.size // 2, 2))
                ones = np.ones(shape=(tmp.shape[0], 1))
                points_ones = np.hstack([tmp, ones])
                tmp = M.dot(points_ones.T).T
                tmp = tmp.reshape(tmp.size)
                tmp = tmp / size * 2 - 1
                tmp[tmp < -1] = -1
                tmp[tmp > 1] = 1
                y_batch[i] = tmp

            if operation == 2:
                angle = np.arange(5, 25, 5)[random.randint(0, 4)]
                M = cv2.getRotationMatrix2D((size // 2, size // 2), -angle, 1.0)
                X_batch[i, :, :, 0] = cv2.warpAffine(X_batch[i, :, :, 0], M, (size, size))

                tmp = y_batch[i]
                tmp = (tmp + 1) / 2 * size
                tmp = tmp.reshape((tmp.size // 2, 2))
                ones = np.ones(shape=(tmp.shape[0], 1))
                points_ones = np.hstack([tmp, ones])
                tmp = M.dot(points_ones.T).T
                tmp = tmp.reshape(tmp.size)
                tmp = tmp / size * 2 - 1
                tmp[tmp < -1] = -1
                tmp[tmp > 1] = 1
                y_batch[i] = tmp

            if operation == 3:
                tmp = y_batch[i]
                tmp = (tmp + 1) / 2 * size
                tmp = tmp.reshape((tmp.size // 2, 2))
                max_x = tmp[0].max()
                max_y = tmp[1].max()
                min_x = tmp[0].min()
                min_y = tmp[1].min()
                if max_x - min_x < size - 1 and max_y - min_y < size - 1:
                    new_size = random.randint(max(max_x - min_x, max_y - min_y), size - 1)
                    x = random.randint(max(0, max_x - new_size), min_x)
                    y = random.randint(max(0, max_y - new_size), min_y)
                    X_batch[i, :, :, 0] = resize(X_batch[i, y: y + new_size, x: x + new_size, 0], (size, size))
                    tmp[0] -= x
                    tmp[1] -= y
                    tmp = tmp.reshape(tmp.size)
                    tmp = tmp / new_size * 2 - 1
                    y_batch[i] = tmp

        return X_batch, y_batch
