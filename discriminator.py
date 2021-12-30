import inspect
import json

import tensorflow as tf
from tqdm import tqdm

from util.dataset import DatasetManager


class Discriminator:
    def __init__(self, config_file=None, custom_model=None, preprocess_input=None, decode_predictions=None):
        if config_file:
            json_file = open(config_file, "r")

            self.config = json.load(json_file)

            input_shape = (self.config["image_width"], self.config["image_height"], self.config["image_channels"])

            weights = self.config["weights"]
            classes = self.config["classes"]

            module_name = self.config["module"]
            model_name = self.config["model"]

            if not module_name == "custom":
                available_modules = dict(inspect.getmembers(tf.keras.applications, inspect.ismodule))
                available_functions = dict(inspect.getmembers(available_modules[module_name], inspect.isfunction))

                # Get the appropriate preprocess input function
                if preprocess_input:
                    self.preprocess_input = preprocess_input
                else:
                    self.preprocess_input = available_functions["preprocess_input"]

                # Get the appropriate decode predictions function
                if decode_predictions:
                    self.decode_predictions = decode_predictions
                else:
                    self.decode_predictions = available_functions["decode_predictions"]

            # Assigning the correct postprocess
            if self.config["mode"] == "caffe":
                self.postprocess = self.caffe_postprocess
            elif self.config["mode"] == "tf":
                self.postprocess = self.tf_postprocess
            elif self.config["mode"] == "torch":
                self.postprocess = self.torch_postprocess

        # Get the appropriate model
        if custom_model:
            self.model = custom_model
            self.preprocess_input = None
            self.decode_predictions = None
        else:
            try:
                model_function = available_functions[model_name]
            except Exception:
                raise ValueError("Discriminator model must be one of " + str(list(available_functions.keys())) + " found '" + model_name + "'")

            self.model = model_function(
                include_top=True,
                weights=weights,
                input_tensor=tf.keras.layers.Input(shape=input_shape),
                classes=classes,
            )

    def predict(self, images):
        verbose = self.config["verbose"]
        workers = self.config["workers"]
        return self.model.predict(images, verbose=verbose, workers=workers, use_multiprocessing=True)

    def clip(self, image):
        new_image = tf.clip_by_value(image, clip_value_min=self.config["clip_min"], clip_value_max=self.config["clip_max"])
        return new_image

    def caffe_postprocess(self, image):
        # Undo zero-centered with respect to the ImageNet dataset (BGR)
        means = [103.939, 116.779, 123.68]
        means_cast = tf.broadcast_to(means, tf.shape(image))
        new_image = tf.math.add(image, means_cast)
        # Change BGR to RGB
        new_image = tf.reverse(new_image, axis=[-1])
        return new_image

    def tf_postprocess(self, image):
        # (-1, 1) to (0, 2)
        new_image = tf.math.add(image, 1)
        # (0, 2) to (0, 255)
        new_image = tf.math.multiply(new_image, 127.5)
        return new_image

    def torch_postprocess(self, image):
        # TODO: Undo the normalize from (0, 1)
        # Normalize each channel with respect to the ImageNet dataset
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        # (0, 1) to (0, 255)
        new_image = tf.math.multiply(image, 255)
        return new_image
