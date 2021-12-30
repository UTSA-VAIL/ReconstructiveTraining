import os

from PIL import Image as pil

def save_image(image, dest):
    image = pil.fromarray(image)
    image.save(dest)