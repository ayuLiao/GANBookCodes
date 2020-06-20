
import os
import numpy as np
import pickle as pkl
from scipy.io import loadmat
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data


class DataLoader(object):
    """
    https://github.com/yunjey/domain-transfer-network
    """

    @staticmethod
    def normalize(images):
        """
        :param images: in range [0, 255]
        :return: in range [-1, 1]
        """
        return 2. * (images / 255.) - 1.


    @staticmethod
    def load_svhn(image_dir, image_file):
        image_dir = os.path.join(image_dir, image_file)
        svhn = loadmat(image_dir)
        images = np.transpose(svhn['X'], [3, 0, 1, 2])
        images = DataLoader.normalize(images)
        labels = svhn['y'].reshape(-1)
        labels[np.where(labels == 10)] = 0
        return images, labels


    @staticmethod
    def resize_images(image_arrays, size=[32, 32]):
        # convert float type to integer
        image_arrays = (image_arrays * 255).astype('uint8')

        resized_image_arrays = np.zeros([image_arrays.shape[0]] + size)
        for i, image_array in enumerate(image_arrays):
            image = Image.fromarray(image_array)
            resized_image = image.resize(size=size, resample=Image.ANTIALIAS)

            resized_image_arrays[i] = np.asarray(resized_image)

        return np.expand_dims(resized_image_arrays, 3)


    @staticmethod
    def prepare_mnist(image_dir, split="train"):
        mnist = input_data.read_data_sets(train_dir=image_dir)
        if split == "train":
            images = DataLoader.resize_images(mnist.train.images.reshape(-1, 28, 28))
            labels = mnist.train.labels
            with open(os.path.join(image_dir, "train.pkl"), "wb") as f:
                pkl.dump((images, labels), f)
        elif split == "test":
            images = DataLoader.resize_images(mnist.test.images.reshape(-1, 28, 28))
            labels = mnist.test.labels
            with open(os.path.join(image_dir, "test.pkl"), "wb") as f:
                pkl.dump((images, labels), f)


    @staticmethod
    def load_mnist(image_dir, split="train"):
        with open(os.path.join(image_dir, split+".pkl"), "rb") as f:
            images, labels = pkl.load(f)
        images = DataLoader.normalize(images)
        return images, labels
