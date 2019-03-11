from PIL import Image
import os
from random import *

IMAGE_DIR = 'data/train/Rococo/'
FINAL_NUM_IMAGES = 8220

image_count = 0
for original_image in os.listdir(IMAGE_DIR):
    image_count += 1

num_images_to_add = FINAL_NUM_IMAGES - image_count

added_images = 0
for original_image in os.listdir(IMAGE_DIR):
    if added_images == num_images_to_add:
        print('Added ' + str(added_images))
        break
    image = Image.open(IMAGE_DIR + original_image)
    
    image_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
    image_name = IMAGE_DIR + original_image + '_flip'
    image_name = image_name.replace('.jpg', '')
    image_name += '.jpg'
    print('Saving flipped ' + image_name + str(added_images))
    image_flip.save(image_name)
    added_images += 1
    if added_images == num_images_to_add:
        print('Added ' + str(added_images))
        break
                    
    
    image_rot_90 = image.rotate(90)
    image_name = IMAGE_DIR + original_image + '_rot'
    image_name = image_name.replace('.jpg', '')
    image_name += '.jpg'
    print('Saving rotated ' + image_name + ' ' + str(added_images))
    image_rot_90.save(image_name)
    added_images += 1
    if added_images == num_images_to_add:
        print('Added ' + str(added_images))
        break
                        
    
    image_rot_180 = image.rotate(180)
    image_name = IMAGE_DIR + original_image + '_rot180'
    image_name = image_name.replace('.jpg', '')
    image_name += '.jpg'
    print('Saving rotated ' + image_name + ' ' + str(added_images))
    image_rot_180.save(image_name)
    added_images += 1
    if added_images == num_images_to_add:
        print('Added ' + str(added_images))
        break
                        

    image_rot_270 = image.rotate(270)
    image_name = IMAGE_DIR + original_image + '_rot270'
    image_name = image_name.replace('.jpg', '')
    image_name += '.jpg'
    print('Saving rotated ' + image_name + ' ' + str(added_images))
    image_rot_270.save(image_name)
    added_images += 1
