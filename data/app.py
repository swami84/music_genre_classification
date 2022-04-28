import streamlit as st
import numpy as np
from pydub import AudioSegment
from presets import Preset
import librosa as librosa
import librosa.display
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input,Conv2D, BatchNormalization, Dense, LSTM,MaxPooling2D
from keras.layers import Reshape, Bidirectional, LSTM,Flatten, Dropout, Activation


def create_model(input_shape=(128, 2881, 1), num_classes=11, model_type='CNN', compile_model=False):
    model = keras.Sequential()

    def step(i, dim, pad, model):
        if i == 0:
            model.add(Conv2D(dim, kernel_size=(3, 3), input_shape=input_shape, name='First_Convolution'))
        else:
            model.add(Conv2D(dim, kernel_size=(3, 3)))

        model.add(BatchNormalization(axis=3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pad, padding='same'))
        model.add(Dropout(0.1))

        return model

    layer_dims = [64, 128, 128, 128]
    pads = [(2, 2), (3, 3), (4, 4), (4, 4)]
    for (i, dim), pad in zip(enumerate(layer_dims), pads):
        model = step(i, dim, pad, model)
    if model_type == 'CNN':
        model.add(Flatten())
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
    if model_type == 'CRNN':
        fin_layer_shape = model.layers[-1].output_shape
        model.add(Reshape((fin_layer_shape[3], fin_layer_shape[2])))
        model.add(Bidirectional(LSTM(128, input_shape=(1, 128, 30), return_sequences=True, )))
        model.add(Bidirectional(LSTM(64, input_shape=(1, 128, 30), return_sequences=False)))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
    if compile_model:
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    return model

