# -*- coding: utf-8 -*-
"""
Created on Sun May 13 23:38:28 2018

@author: 
"""

# Load pickled data
import numpy as np
#import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import glob
import cv2
# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Lambda, Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D
#from keras.layers.convolutional import Con
from keras import optimizers
#from keras.callbacks import ModelCheckpoint
from keras.models import load_model

PICS_DIR = '..\\windows_sim\\data\\'
# the frame size after cropping the unimportant data
I_WIDTH = 320
I_HEIGHT = 75
load_previous_weights = '' #model-till-bridge.h5' use to initialize weights

#model exactly as described in nvidia paper
def bc_model(new_model):
    if(new_model==False): #start with a previously trained model
        return load_model(load_previous_weights)
    
    model = Sequential()
    # preprocess data, normalization. the input format is BGR, 3 ch
    model.add(Lambda(lambda x: x/128.0-1.0, input_shape=(I_HEIGHT, I_WIDTH, 3)))
    model.add(Conv2D(24, kernel_size=(5, 5), activation='relu', strides=(2,2)))
    model.add(Conv2D(36, kernel_size=(5, 5), activation='relu', strides=(2,2)))
    model.add(Conv2D(48, kernel_size=(5, 5), activation='relu', strides=(2,2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))
    
    # printing the info
    for layer in model.layers:
        print(layer.get_output_at(0).get_shape().as_list())

    return model

#loading the data from the csv to memory    
def load_pics():
    files = glob.glob(PICS_DIR+'*.csv')
    data = pd.DataFrame()
    
    for file in files:
        tmpdata = pd.read_csv(file, header=None, 
                      names=['center', 'left', 'right', 'steering', 'xx', 'xxx', 'speed'])    
        if int(file[-6:-4])>0:
            for i in range(9):
                data = pd.concat([data, tmpdata])
        data = pd.concat([data, tmpdata])

    #take 3 cameras and steering angle
    X = data[['center', 'left', 'right']].values
    y = data['steering'].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
    X_train, y_train = shuffle(X_train, y_train)
    X_valid, y_valid = shuffle(X_valid, y_valid)
    
    return X_train, y_train, X_valid, y_valid

#remove the unimportant backgroud that might just confuse the model. also, less params to do SGD
def crop(image):
    return image[60:-25, :, :] # remove the sky and the car front

#data augmentation
def flip(image, steering_angle):
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle

STEERING_FACTOR = 0.2
def get_image(data_dir, center, left, right, steering_angle):
    choice = np.random.choice(3)
    if choice == 0:
        return cv2.imread(data_dir+left), steering_angle + STEERING_FACTOR
    elif choice == 1:
        return cv2.imread(data_dir+right), steering_angle - STEERING_FACTOR
    return cv2.imread(data_dir+center), steering_angle

#add randomness by moving the picture around and adjust the steering angle
def translate(image, steering_angle, range_x, range_y):
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle

#add randomness to the brightness of the pic
def brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

#add richness to the recorded data
def augument(data_dir, center, left, right, steering_angle, range_x=100, range_y=10):
    image, steering_angle = get_image(data_dir, center, left, right, steering_angle)
    image, steering_angle = flip(image, steering_angle)
    image, steering_angle = translate(image, steering_angle, range_x, range_y)
    image = brightness(image)
    return image, steering_angle

#helper function to not load all pics to memory at once
def generator(data_dir, image_paths, steering_angles, batch_size, is_training):
    images = np.empty([batch_size, I_HEIGHT, I_WIDTH, 3])
    steers = np.empty(batch_size)
    while 1:
        i = 0
        for index in range(len(image_paths)):
            center, left, right = image_paths[index]
            #preprocessing because I moved the picture folders
            center = center.split(sep="\\")[-1]
            left = left.split(sep="\\")[-1]
            right = right.split(sep="\\")[-1]
            steering_angle = steering_angles[index]
            # augumentation - adding some randomness to the data
            if is_training and np.random.rand() < 0.5:
                image, steering_angle = augument(data_dir, center, left, right, steering_angle)
            else:
                image = cv2.imread(data_dir + center) 
            # add the image and angle to the batch
            images[i] = crop(image)
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers

if __name__ == '__main__':

    X_train, y_train, X_valid, y_valid = load_pics()
    model = bc_model(new_model=(load_previous_weights==''))
    
    opt = optimizers.Adam(lr=1e-5)
    model.compile(loss='mean_squared_error', optimizer=opt)
    
    BATCH_SIZE = 40
    
    #point to where the pics are
    train_gen = generator(PICS_DIR+'combined\\', X_train, y_train, BATCH_SIZE, True)
    valid_gen = generator(PICS_DIR+'combined\\', X_valid, y_valid, BATCH_SIZE, False)
    
    #start training
    history_object = model.fit_generator(train_gen, steps_per_epoch=np.floor(len(X_train)/BATCH_SIZE), 
                        epochs=10, max_queue_size=1,
                        validation_data=valid_gen,
                        validation_steps=np.floor(len(X_valid)/BATCH_SIZE),
                        verbose=1)
    #save the last set of weights
    model.save('my_model.h5')

    from matplotlib import pyplot as plt
    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
