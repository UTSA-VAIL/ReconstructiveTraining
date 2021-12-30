import os

import tensorflow as tf

import util


class DatasetManager:
    def __init__(self, dir_path, shape):
        file_list = sorted([os.path.join(dir_path, filename) for filename in os.listdir(dir_path)])
        self.ds = tf.data.Dataset.from_tensor_slices(file_list)
        self.shape = shape
        self.read_decode()
        # self.read_decode_v2()
        self.set_shape_cast(self.shape)

    def __call__(self):
        return self.ds

    def __str__(self):
        return self.ds.__str__()

    def map(self, func):
        self.ds = self.ds.map(func, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def batch(self, size):
        self.ds = self.ds.batch(size)

    def prefetch(self):
        self.ds = self.ds.prefetch(tf.data.experimental.AUTOTUNE)

    def set_shape(self, image, shape_size):
        image.set_shape(shape_size)
        return image

    def set_shape_cast(self, shape):
        self.map(lambda image: self.set_shape(image, shape))
        self.map(lambda x: tf.cast(x, tf.float32))

    def read_decode(self):
        self.map(lambda filename: tf.io.read_file(filename))
        self.map(lambda contents: tf.io.decode_image(contents, channels=3))

    def read_decode_v2(self):
        self.map(lambda filename: tf.io.read_file(filename))
        self.map(lambda contents: tf.io.decode_image(contents, channels=3, dtype=tf.dtypes.uint8, expand_animations=False)) # Image in 0-255 format
        self.map(lambda image:  tf.keras.preprocessing.image.smart_resize(image, (256, 256)))
        self.map(lambda image: tf.image.central_crop(image, (self.shape[0]/256)))

    def prep(self, mode="tf"):
        if mode == "tf":
            # sets pixle range to: (-1, 1)
            center = 255.0 / 2.0
            self.map(lambda image: tf.math.divide(image, center))
            self.map(lambda image: tf.math.subtract(image, 1))

        elif mode == "caffe":
            # sets pixle range to: (0, 255) and RGB to BGR
            self.map(lambda image: tf.reverse(image, axis=[-1]))
            # zero-centered with respect to the ImageNet dataset (BGR)
            self.mean = [103.939, 116.779, 123.68]
            means_cast = tf.broadcast_to(self.means, self.shape)
            self.map(lambda image: tf.math.subtract(image, means_cast))

        elif mode == "torch":
            # sets pixle range to: (0, 1)
            self.map(lambda image: tf.math.divide(image, 255.0))
            # normalize
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
            self.axis = -3

        else:
            raise NotImplementedError
