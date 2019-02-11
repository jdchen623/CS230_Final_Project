import tensorflow as tf
import os
import numpy as np
from PIL import Image

"""
Requires images as .jpg files in ./data/test/data

Params: resizeHeight, resizeWidth: dimensions in pixels to resize each image to
Returns: dict{filename: imageArray}
"""
def images_as_arrays(resizeHeight, resizeWidth):
    image_dict = {}
    for filename in os.listdir("./data/test_data"):
        if filename.endswith(".jpg"):
            filepath = './data/test_data/' + filename
            image = np.array(Image.open(filepath))
            image_resized = np.resize(image, (resizeHeight, resizeWidth, 3)).flatten()
            image_dict[filename] = image_resized
    return image_dict

"""
Requires images as .jpg files in ./data/test/data

Params: resizeHeight, resizeWidth: dimensions in pixels to resize each image to
Returns: dict{filename: imageArray} 
"""
def images_as_tensors(resizeHeight, resizeWidth):
    image_dict = {}
    for filename in os.listdir('./data/test_data/'):
        if filename.endswith(".jpg"):
            filepath = './data/test_data/' + filename
            image = np.array(Image.open(filepath))
            image_resized = np.resize(image, (resizeHeight, resizeWidth, 3)).flatten()
            image_tensor = tf.convert_to_tensor(image_resized, dtype=tf.float32)
            image_dict[filename] = image_tensor
    return image_dict
        
print(images_as_arrays(200, 200))
