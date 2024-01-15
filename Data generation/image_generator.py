"""
This file contains the code to generate the distance maps and H-stains for the images.
The distance maps are generated using the distance map generator model based on a tensorflow implementation of the U-Net architecture.
The H-stains are generated using a stain normilization algorithm.
Fill in the variables below to run the code.
The folders in the destination path will not be created automatically, so make sure they exist.
"""
# The path to the images to be transformed.
image_path = r'/home/tijmenhijzelaar/TUE_project_2023/TUE_tijmen_new/' 
# The path to the directory where the transformed images will be saved.
dest_path_dmap = r'/home/tijmenhijzelaar/TUE_project_2023/Dmap_old/'
dest_path_hstain = r'/home/tijmenhijzelaar/TUE_project_2023/H-stains/'
# The path to the weights of the distance map generator model.
checkpoint_path = r"./Distance Map Models/Model2.ckpt"

# Whether to save the distance maps or H-stains seperately or not.
save_dmap = True
save_hstain = True
# Whether to normalize the distance maps or not.
d_norm = True



import tensorflow as tf
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from keras.preprocessing import image
import numpy as np
from normalizeStaining import normalizeStaining as separate_stain
#import matplotlib.pyplot as plt
import glob
import os
import cv2
#######################################################################################
def mse_score(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)
#######################################################################################################
def distance_unet(IMG_CHANNELS, LearnRate):
    inputs = Input((None, None, IMG_CHANNELS))
    s = Lambda(lambda x: x / 255)(inputs)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (s)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9)

    outputs = Conv2D(1, (1, 1), activation='linear') (c9)

    model_dis = models.Model(inputs=[inputs], outputs=[outputs])
    model_dis.compile(optimizer = Adam(lr= LearnRate), loss='mean_squared_error', metrics=[mse_score])
    return model_dis



def load_image(img_path):

    img = image.load_img(img_path, target_size=(1024,1024))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)

    return img_tensor



def dmap(img, model):
  # Create the predicted distance map and remove any negative values
  # by shifting all values with the absolute value of the lowest data entry
  return np.clip(model.predict(img),0,None)


def apply_dmap_and_save(img, model, dir, name, save_dmap = False, d_norm=False):
  d = dmap(img,model)
  #norm_img, stain,   = separate_stain(img[0])[:2]
  #stain =tf.image.rgb_to_grayscale(stain)
  
  if d_norm:
    d *= 255.0/np.max(d)
    
  new_img = np.concatenate((img,d),axis=-1)[0]

  image.save_img(dir+name, new_img)
  if save_dmap:
    image.save_img(dir+"dmap "+name, d[0])

def transform_data_dmap(path, dest_path, Model, save_dmap = False, d_norm = False):
  for subdir, _, _ in os.walk(path):
    # Skip parent directory
    if subdir == path:
                continue
    # Find images in the subdirectory
    base_name_subdir = os.path.basename(subdir)
    for file in glob.glob(subdir +"**/*.png", recursive=True):
      im = load_image(file)
      base_name_file = os.path.basename(file)
      apply_dmap_and_save(im, Model, dest_path+base_name_subdir+'/', base_name_file, save_dmap=save_dmap, d_norm = d_norm)


def apply_h_stain_and_save(img, dir, name, save_hstain= False):
  # Apply stain normalization
  norm_img, stain,   = separate_stain(img[0])[:2]
  stain =tf.image.rgb_to_grayscale(stain)
  # Apply the H-stain to the image 
  new_img = np.concatenate((img[0],stain),axis=-1)

  image.save_img(dir+name, new_img)

  if save_hstain:
    image.save_img(dir+"H-stain "+name, stain)

def transform_data_hstain(path, dest_path, save_hstain = False):
  for subdir, _, _ in os.walk(path):
    # Skip parent directory
    if subdir == path:
                continue
    # Find images in the subdirectory
    base_name_subdir = os.path.basename(subdir)
    for file in glob.glob(subdir +"**/*.png", recursive=True):
      im = load_image(file)
      base_name_file = os.path.basename(file)
      apply_h_stain_and_save(im, dest_path+base_name_subdir+'/', base_name_file, save_hstain=save_hstain)
  
if __name__ == "__main__":
  # Initiate model and load the weights
  Model = distance_unet(3, 0.001)
  Model.load_weights(checkpoint_path)
  # Transform the data
  transform_data_dmap(image_path, dest_path_dmap, Model, save_dmap = save_dmap, d_norm = d_norm)
  transform_data_hstain(image_path, dest_path_hstain, save_hstain = save_hstain)