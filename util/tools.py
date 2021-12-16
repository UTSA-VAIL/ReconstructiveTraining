import random

import numpy as np
import tensorflow as tf


def setup_missing_squares(image_blocks, indicies):
    new_image_blocks = image_blocks
    # Set the selected squares to zero resulting in black squares
    for i in indicies:
        new_image_blocks[i] = np.zeros((image_blocks[i].shape))
    
    # Return the new image with the missing squares
    return new_image_blocks

def setup_rbg2gray(image_blocks, indicies):
    new_image_blocks = image_blocks
    # Determine the weights for converting to grayscale.
    rgb_weights = [0.2989, 0.5870, 0.1140]
    for i in indicies:
        # Calculate the grayscale value
        gray_values = (np.dot(new_image_blocks[i][...,:3], rgb_weights)).astype(np.uint8)

        # Assign the grayscale value to all 3 channels
        new_image_blocks[i][...,0] = gray_values
        new_image_blocks[i][...,1] = gray_values
        new_image_blocks[i][...,2] = gray_values
    
    # Return the new image with grayscale squares
    return new_image_blocks

def setup_jpegcompression(image, indicies, compression_quality=10):
    # Setup some tensorflow variables
    image_shape = tf.shape(image)
    h, w, c = image_shape[0], image_shape[1], image_shape[2],

    # Split the image into tiles
    tile_rows = tf.reshape(image, [h, -1, 28, c])
    serial_tiles = tf.transpose(tile_rows, [1, 0, 2, 3])
    image_blocks = tf.reshape(serial_tiles, [-1, 28, 28, 3])
  
    # Encode and decode the image
    for i, block in enumerate(image_blocks):
        
        if i not in indicies:
            quality = 100
        else:
            quality = compression_quality
        
        new_block = tf.image.decode_jpeg(
            tf.image.encode_jpeg(
                tf.cast(block, dtype=tf.uint8), 
                format='rgb', 
                quality=quality
            ),
            channels=3
        )

        new_block = tf.expand_dims(new_block, axis=0)

        if i == 0:
            new_image_blocks = new_block
        else:
            new_image_blocks = tf.concat([new_image_blocks, new_block], axis=0)

    # Reconstruct the image using the new tiles
    seriaized_tiles = tf.reshape(new_image_blocks, [-1, h, 28, c])
    rowwise_tiles = tf.transpose(seriaized_tiles, [1, 0, 2, 3])
    new_image = tf.reshape(rowwise_tiles, [h, w, c])

    # Return the reconstructed image with compressed squares
    return new_image