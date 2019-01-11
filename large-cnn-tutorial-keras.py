# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 16:35:58 2019

@author: Pensée Artificielle
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
K.set_image_dim_ordering('th')

import preparedata as pr
import cnnutils as cu

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# import MNIST dataset
(X_train, y_train), (X_test, y_test), num_classes = pr.get_and_prepare_data_mnist()

# define the larger model
def large_model():
    # create model
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# build the model
model = large_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)

# Evaluate the model
cu.print_model_error_rate(model, X_test, y_test)
# Save the model
cu.save_keras_model(model, "save_model/large_model_cnn")