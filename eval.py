import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np

import tensorflow as tf

from util.dataset import DatasetManager
from discriminator import Discriminator
from util.stats import get_top_results


if __name__ == "__main__":
    # Start the argument parser
    parser = argparse.ArgumentParser(description="Run Evaluations")

    # User input
    parser.add_argument("--input_dir", required=True, type=str, help="Directory to input images")
    parser.add_argument("--image_size", required=True, type=int, nargs=3, metavar=("HEIGHT", "WIDTH", "COLOR"), help="Size of a single image")
    parser.add_argument("--labels_file", required=True, type=str, help="Path to the labels file")
    parser.add_argument("--num_classes", required=True, type=int, help="Number of classes")
    parser.add_argument("--discriminator_config_file", required=True, type=str, help="path to the discriminator's config file")

    args = parser.parse_args()

    # Labels
    labels = np.loadtxt(args.labels_file)

    ORIG_DATASET_SIZE = len(labels)
    TRAIN_SIZE = int(0.90 * ORIG_DATASET_SIZE)
    VAL_SIZE = int(0.05 * ORIG_DATASET_SIZE)
    TEST_SIZE = int(0.05 * ORIG_DATASET_SIZE)

    with tf.distribute.MirroredStrategy().scope():
        # Discriminator Model
        discriminator = Discriminator(config_file=args.discriminator_config_file)


    # print("Evaluating:", args.input_dir)
    # Load in the testing images
    image_manager = DatasetManager(args.input_dir, args.image_size)
    image_manager.map(discriminator.preprocess_input)
    test_images_ds = image_manager.ds
    test_images_ds = test_images_ds.prefetch(tf.data.experimental.AUTOTUNE)
    test_images_ds = test_images_ds.batch(500)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    test_images_ds = test_images_ds.with_options(options)
   
    # Results
    y_pred = discriminator.predict(test_images_ds)
    results = get_top_results(preds=y_pred, labels=labels)
    print(np.asarray(results) / 100)
    mce_results = results[0] / 100
    print(1 - mce_results)