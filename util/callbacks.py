import os

import numpy as np
import tensorflow as tf

import util

class GeneratorOutputObserver(tf.keras.callbacks.Callback):
    """"
    callback to observe the output of the network
    """

    def __init__(self, model_name, origs, advs, dir_path, title_list, num_samples=20):
        self.model_name = model_name
        self.origs = np.asarray(list(origs.take(num_samples).as_numpy_iterator()))
        self.origs_post = ((self.origs + 1) * (255/2)).astype(np.uint8)
        self.advs = np.asarray(list(advs.take(num_samples).as_numpy_iterator()))
        self.advs_post = ((self.advs + 1) * (255/2)).astype(np.uint8)
        self.dir_path = dir_path
        self.title_list = title_list

    def on_epoch_end(self, epoch, logs={}):
        print("Saving samples in", self.dir_path)
        gen_output = self.model.predict(self.advs)
        # gen_output = gen_output.numpy()
        gen_output = ((gen_output + 1) * (255/2)).astype(np.uint8)

        for j, (orig, adv, gen) in enumerate(zip(self.origs_post, self.advs_post, gen_output)):
            filename = os.path.join(self.dir_path, "{0:03d}_{1:03d}.png".format(epoch+1, j))
            util.plot_summary(
                filename,
                self.title_list,
                [orig, adv, gen],
                self.model_name,
            )

class GANOutputObserver(tf.keras.callbacks.Callback):
    """"
    callback to observe the output of the network
    """

    def __init__(self, model_name, origs, advs, dir_path, title_list, num_samples=20):
        self.model_name = model_name
        self.origs = np.asarray(list(origs.take(num_samples).as_numpy_iterator()))
        self.origs_post = ((self.origs + 1) * (255/2)).astype(np.uint8)
        self.advs = np.asarray(list(advs.take(num_samples).as_numpy_iterator()))
        self.advs_post = ((self.advs + 1) * (255/2)).astype(np.uint8)
        self.dir_path = dir_path
        self.title_list = title_list

    def on_epoch_end(self, epoch, logs={}):
        print("Saving samples in", self.dir_path)
        gen_output = self.model.predict(self.advs)
        gen_output = ((gen_output[0] + 1) * (255/2)).astype(np.uint8)

        for j, (orig, adv, gen) in enumerate(zip(self.origs_post, self.advs_post, gen_output)):
            filename = os.path.join(self.dir_path, "{0:03d}_{1:03d}.png".format(epoch+1, j))
            util.plot_summary(
                filename,
                self.title_list,
                [orig, adv, gen],
                self.model_name,
            )


class GANModelCheckpoint(tf.keras.callbacks.Callback):
    """
        Used to save weights for the GAN.
        Only generator weights will be saved to checkpoint file as the discriminator is not important.

        Saves the best val_loss
    """

    def __init__(self, generator_model, output_path, monitor):
        self.generator_model = generator_model
        self.output_path = output_path
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs={}):
        current_loss = logs.get(self.monitor)
        self.generator_model.save_weights(os.path.join(self.output_path, "{0:03d}--{1:010f}".format(epoch, current_loss)))



