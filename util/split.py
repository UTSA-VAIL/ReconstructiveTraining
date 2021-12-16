import numpy as np


def split_image(image, nrows, ncols):
    R = image.shape[0]//nrows
    C = image.shape[1]//ncols

    tiles = [image[x:x+R,y:y+C] for x in range(0,image.shape[0],R) for y in range(0,image.shape[1],C)]
    return np.array(tiles)

def rebuild_split_image(image_blocks, nrows, ncols):
    final_image = np.empty([224,224,3], dtype=np.uint8)
    row_size = image_blocks[0].shape[0]
    col_size = image_blocks[0].shape[1]
    for x in range(nrows):
        for y in range(ncols):
            final_image[x*row_size:(x+1)*row_size,y*col_size:(y+1)*col_size] = image_blocks[x*ncols + y]

    return np.array(final_image)