import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from util.callbacks import *
from util.dataset import DatasetManager
from util.loss import Loss


class Autoencoder:
    def __init__(self, input_shape):
        # Create the model
        self.model = self.create_model(input_shape)

        # Give the model a name
        self.model._name = "FCA"

    def conv_layer(self, filters, kernel_size, prev_layer, strides=1):
        layer = tf.keras.layers.Conv2D(filters=filters, padding="same", kernel_size=kernel_size, strides=strides)(prev_layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.Activation('relu')(layer)
        return layer

    def conv_transpose_layer(self, filters, kernel_size, prev_layer, strides=1):
        layer = tf.keras.layers.Conv2DTranspose(filters=filters, padding="same", kernel_size=kernel_size, strides=strides)(prev_layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.Activation('relu')(layer)
        return layer

    def create_model(self, shape):
        # Determine the base number of filters
        num_filters = 48

        # Input layer
        input_layer = tf.keras.layers.Input(shape=shape)
            
        # Encoder - Block 1
        encoding_blk1_layer = self.conv_layer(filters=num_filters, kernel_size=3, prev_layer=input_layer)
        encoding_blk1_layer = self.conv_layer(filters=num_filters, kernel_size=3, prev_layer=encoding_blk1_layer)
        encoding_blk1_layer = self.conv_layer(filters=num_filters, kernel_size=3, prev_layer=encoding_blk1_layer, strides=2)

        # Encoder - Block 2
        encoding_blk2_layer = self.conv_layer(filters=num_filters*2, kernel_size=3, prev_layer=encoding_blk1_layer)
        encoding_blk2_layer = self.conv_layer(filters=num_filters*2, kernel_size=3, prev_layer=encoding_blk2_layer)
        encoding_blk2_layer = self.conv_layer(filters=num_filters//2, kernel_size=3, prev_layer=encoding_blk2_layer, strides=2)

        # Encoder - Block 3
        encoding_blk3_layer = self.conv_layer(filters=num_filters*4, kernel_size=3, prev_layer=encoding_blk2_layer)
        encoding_blk3_layer = self.conv_layer(filters=num_filters*4, kernel_size=3, prev_layer=encoding_blk3_layer)
        encoding_blk3_layer = self.conv_layer(filters=num_filters*4, kernel_size=3, prev_layer=encoding_blk3_layer, strides=2)

        # Decoder - Block 3
        decoding_blk3_layer = self.conv_transpose_layer(filters=num_filters//4, kernel_size=3, prev_layer=encoding_blk3_layer, strides=2)
        decoding_blk3_layer = self.conv_layer(filters=num_filters*4, kernel_size=3, prev_layer=decoding_blk3_layer)
        decoding_blk3_layer = self.conv_layer(filters=num_filters*4, kernel_size=3, prev_layer=decoding_blk3_layer)

        # Concatenate the output of encoder block 2 with output decoding block 3 for detail
        # concatenate_layer_1 = tf.keras.layers.Concatenate()([decoding_blk3_layer, encoding_blk2_layer])

        # Decoder - Block 2
        decoding_blk2_layer = self.conv_transpose_layer(filters=num_filters//2, kernel_size=3, prev_layer=decoding_blk3_layer, strides=2)
        decoding_blk2_layer = self.conv_layer(filters=num_filters*2, kernel_size=3, prev_layer=decoding_blk2_layer)
        decoding_blk2_layer = self.conv_layer(filters=num_filters*2, kernel_size=3, prev_layer=decoding_blk2_layer)

        # Concatenate the output of encoder block 1 with output decoding block 2 for detail
        concatenate_layer_2 = tf.keras.layers.Concatenate()([decoding_blk2_layer, encoding_blk1_layer])

        # Decoder - Block 1
        decoding_blk1_layer = self.conv_transpose_layer(filters=num_filters, kernel_size=3, prev_layer=concatenate_layer_2, strides=2)
        decoding_blk1_layer = self.conv_layer(filters=num_filters, kernel_size=3, prev_layer=decoding_blk1_layer)
        decoding_blk1_layer = self.conv_layer(filters=num_filters, kernel_size=3, prev_layer=decoding_blk1_layer)

        # Image Output Layer
        output_layer = tf.keras.layers.Conv2D(filters=3, padding="same", kernel_size=3, activation="tanh")(decoding_blk1_layer)

        # Create the model
        return tf.keras.models.Model(inputs=input_layer, outputs=output_layer)


    def post_process(self, images):
        # Change the images from a (-1,1) format to a (0,255) format
        images = tf.math.add(images, 1)
        images = tf.math.multiply(images, (127.5))
        return images

    def black_and_white_change(self, images):
        grey_scale_images = tf.image.rgb_to_grayscale(images)
        return grey_scale_images


if __name__ == "__main__":
    # Start the argument parser
    parser = argparse.ArgumentParser(description='Run Generator')

    # User input
    parser.add_argument('--original_input_dir', required=True, type=str, help='Directory to original/preprocessed images')
    parser.add_argument('--attacked_input_dir', required=True, type=str, help='Directory to attacked images')
    parser.add_argument('--weights_output_dir', required=True, type=str, help='Directory to output weights')
    parser.add_argument('--logs_output_dir', required=True, type=str, help='Directory to output logs')
    parser.add_argument('--image_size', required=True, type=int, nargs=3, metavar=('HEIGHT', 'WIDTH', 'COLOR'), help='Size of a single image')
    parser.add_argument('--batch_size', required=False, default=125, type=int, help='Batch size to train on')
    parser.add_argument('--epochs', required=False, default=50, type=int, help='Number of epochs to train')
    
    # Parser through the arguments
    args = parser.parse_args()

    # Create the missing directories if needed
    if not os.path.exists(args.weights_output_dir):
        os.makedirs(args.weights_output_dir)

    if not os.path.exists(args.logs_output_dir):
        os.makedirs(args.logs_output_dir)

    # Handle multiple GPU support
    with tf.distribute.MirroredStrategy().scope():
        # Start the generator
        generator = Autoencoder(input_shape=args.image_size)
        model = generator.model

        init_epoch = 0
        
        # Resume from a previous checkpoint if avaliable
        ckpt = tf.train.latest_checkpoint(args.weights_output_dir)
        if ckpt != None:
            print("Loading Checkpoint:", ckpt)
            init_epoch = int(os.path.basename(ckpt).split("--")[0])
            model.load_weights(ckpt)

        # OPTIMIZER
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # LOSS
        psnr = Loss(loss_names=['psnr'], psnr_max_value=2.0)
        rmse = Loss(loss_names=['rmse'], rmse_reduction=tf.keras.losses.Reduction.SUM)
        l2 = Loss(loss_names=['l2dist'])
        loss_function = Loss(
            loss_names=['psnr', 'rmse'], 
            loss_weights=[0.01, 1.0], 
            psnr_max_value=2.0,
            rmse_reduction=tf.keras.losses.Reduction.SUM
        )

        # Model compile
        model.compile(
            loss=loss_function,
            metrics=[l2, rmse, psnr],
            optimizer=optimizer
        )

        callbacks_list = [
            tf.keras.callbacks.CSVLogger(
                os.path.join(args.logs_output_dir, "training.csv"), 
                append=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(args.weights_output_dir, "{epoch:03d}--{val_loss:0.4f}"),
                monitor='val_loss',
                verbose=1,
                save_best_only=False,
                save_weights_only=True,
                mode='min',
                save_freq='epoch',
            ),
            tf.keras.callbacks.TerminateOnNaN(),
        ]

    print(generator.model.summary())
    
    # Get the sizes of the different sets
    ORIG_DATASET_SIZE = len(os.listdir(args.original_input_dir))
    ATK_DATASET_SIZE = len(os.listdir(args.attacked_input_dir))
    TRAIN_SIZE = int(0.90 * ORIG_DATASET_SIZE)
    VAL_SIZE = int(0.05 * ORIG_DATASET_SIZE)
    TEST_SIZE = int(0.05 * ORIG_DATASET_SIZE)

    print("Original Dataset Size: {:6d}".format(ORIG_DATASET_SIZE))
    print("Attack Dataset Size:   {:6d}".format(ATK_DATASET_SIZE))
    print("Training Size:         {:6d}".format(TRAIN_SIZE))
    print("Validation Size:       {:6d}".format(VAL_SIZE))
    print("Testing Size:          {:6d}".format(TEST_SIZE))

    # Load in the original images
    orig_image_manager = DatasetManager(args.original_input_dir, args.image_size)
    orig_image_manager.prep()
    print("Finished loading original images")

    # Load in the adversarial images
    adv_image_manager = DatasetManager(args.attacked_input_dir, args.image_size)
    adv_image_manager.prep()
    print("Finished loading adversarial images")
 
    # Zip the images and split into training and validation sets
    full_ds = tf.data.Dataset.zip((adv_image_manager.ds, orig_image_manager.ds))
    full_ds = full_ds.prefetch(tf.data.experimental.AUTOTUNE)
    train_data_ds = full_ds.take(TRAIN_SIZE)
    val_test_data_ds = full_ds.skip(TRAIN_SIZE)
    val_data_ds = val_test_data_ds.take(VAL_SIZE)
    
    # Batch the data
    train_data_ds = train_data_ds.shuffle(TRAIN_SIZE)
    train_data_ds = train_data_ds.batch(args.batch_size)
    train_data_ds = train_data_ds.prefetch(tf.data.experimental.AUTOTUNE)
    
    val_data_ds = val_data_ds.shuffle(VAL_SIZE)
    val_data_ds = val_data_ds.batch(args.batch_size)
    val_data_ds = val_data_ds.prefetch(tf.data.experimental.AUTOTUNE)
    print("Finished batching images")

    # Run the model
    model.fit(
        train_data_ds,
        epochs=args.epochs,
        initial_epoch=init_epoch,
        steps_per_epoch=TRAIN_SIZE // args.batch_size,
        verbose=1,
        validation_data=val_data_ds,
        validation_steps=VAL_SIZE // args.batch_size,
        workers=36,
        use_multiprocessing=True,
        callbacks=callbacks_list,
    )