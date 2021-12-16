import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys

import numpy as np
import tensorflow as tf

from discriminator import Discriminator
from generator import Autoencoder
from util.callbacks import GANModelCheckpoint
from util.dataset import DatasetManager
from util.loss import Loss


class GAN:
    def __init__(self, shape, generator, descriminator):
        # Set the inital paramters
        self.generator = generator
        self.descriminator = descriminator

        # Create the model
        self.model = self.create_model(shape)
    
    def create_model(self, shape):
        # Get the input for the generator
        gan_input = tf.keras.Input(shape=shape)

        # Get the output from the generator
        generator_output = self.generator.model(gan_input)

        # Get the format from -1 to 1 to 0 to 255
        discriminator_input = self.generator.post_process(generator_output)

        # Check if the preprocess exists
        if self.descriminator.preprocess_input is not None:
            # Apply the descriminator preprocess to the generator output
            descriminator_input_pp = self.descriminator.preprocess_input(discriminator_input)

        # Get the output from the descriminator
        descriminator_output = self.descriminator.model(descriminator_input_pp)

        # Create the GAN model
        return tf.keras.Model(inputs=gan_input, outputs=[generator_output, descriminator_output])
    

    def generate(self, images):
        imgs = self.generator.generate(images)
        return imgs


if __name__ == '__main__':
    # Start the argument parser
    parser = argparse.ArgumentParser(description='Run Generator')

    # User input
    parser.add_argument('--original_input_dir', required=True, type=str, help='Directory to original/preprocessed images')
    parser.add_argument('--attacked_input_dir', required=True, type=str, help='Directory to attacked images')
    parser.add_argument('--generator_input_dir', required=True, type=str, help='Directory to generator starting weights')
    parser.add_argument('--weights_output_dir', required=True, type=str, help='Directory to output weights')
    parser.add_argument('--logs_output_dir', required=True, type=str, help='Directory to output logs')
    parser.add_argument('--discriminator_config_file', required=True, type=str, help="path to the discriminator's config file")
    parser.add_argument('--labels_file', required=True, type=str, help='Labels to images')
    parser.add_argument('--num_classes', required=True, type=int, help="Number of classes")
    parser.add_argument('--image_size', required=True, type=int, nargs=3, metavar=('HEIGHT', 'WIDTH', 'COLOR'), help='Size of a single image')
    parser.add_argument('--batch_size', required=False, default=125, type=int, help='Batch size to train on')
    parser.add_argument('--epochs', required=False, default=150, type=int, help='Number of epochs to train')

    # Parse the arguments
    args = parser.parse_args()

    # Create the directories if they don't already exist
    if not os.path.exists(args.weights_output_dir):
        os.makedirs(args.weights_output_dir)

    if not os.path.exists(args.logs_output_dir):
        os.makedirs(args.logs_output_dir)

    # Handle multiple GPU support
    with tf.distribute.MirroredStrategy().scope():
        # Start the generator
        generator = Autoencoder(input_shape=args.image_size)

        # Load the latest pretrained model' weights
        generator_ckpt = tf.train.latest_checkpoint(args.generator_input_dir)
        if generator_ckpt != None:
            print("Loading Checkpoint:", generator_ckpt)
            generator.model.load_weights(generator_ckpt)

        # Load from a checkpoint if it exists
        init_epoch = 0
        gan_ckpt = tf.train.latest_checkpoint(args.weights_output_dir)
        if gan_ckpt != None:
            print("Loading Checkpoint:", gan_ckpt)
            init_epoch = int(os.path.basename(gan_ckpt).split("--")[0])+1
            generator.model.load_weights(gan_ckpt)
        
        # Start the descriminator
        discriminator = Discriminator(args.discriminator_config_file)

        # Set the descriminator model to not trainable
        discriminator.model.trainable = False

        # Start the GAN
        gan = GAN(shape=args.image_size, generator=generator, descriminator=discriminator)
        model = gan.model

        # Start the optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # Create the xustom combination loss
        psnr = Loss(loss_names=['psnr'], psnr_max_value=255.0)
        rmse = Loss(loss_names=['rmse'], rmse_reduction=tf.keras.losses.Reduction.SUM)
        l2 = Loss(loss_names=['l2dist'])
        loss_function = Loss(
            loss_names=['psnr', 'rmse'], 
            loss_weights=[0.01, 1.0], 
            psnr_max_value=255.0,
            rmse_reduction=tf.keras.losses.Reduction.SUM
        )
        
        # Start the optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # Compile the model
        model.compile(
            loss=[
                loss_function,
                'categorical_crossentropy',
            ],
            loss_weights=[0.005,1.0], 
            metrics=[
                [rmse, psnr],
                'accuracy',
            ],
            optimizer=optimizer
        )

        # CALLBACKS
        callbacks_list = [
            tf.keras.callbacks.CSVLogger(
                os.path.join(args.logs_output_dir, "training.csv"), 
                append=True
            ),
            GANModelCheckpoint(
                generator_model=gan.generator.model,
                output_path=args.weights_output_dir,
                monitor='val_loss',
            ),
            tf.keras.callbacks.TerminateOnNaN(),
        ]
    
    print(gan.model.summary())

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

    # Load in the labels of the file
    labels = np.loadtxt(args.labels_file, dtype=int)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    labels_ds = labels_ds.map(lambda label: tf.one_hot(label, args.num_classes), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Combine original and label into a single label
    combined_labels_ds = tf.data.Dataset.zip((orig_image_manager.ds, labels_ds))

    # Zip the data and split into training and validation sets
    full_ds = tf.data.Dataset.zip((adv_image_manager.ds, combined_labels_ds))
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