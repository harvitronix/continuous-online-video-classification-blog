"""
Combining CNN and RNN into a CRNN, using Keras' TimeDistributed wrapper.
"""
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from scipy.misc import imread
from collections import deque
import numpy as np
import sys
import pickle

def get_model(input_shape):
    """Build the network.
    Starting version from:
    https://github.com/udacity/self-driving-car/blob/master/
        steering-models/community-models/chauffeur/models.py
    """
    model = Sequential()
    model.add(TimeDistributed(Convolution2D(24, 5, 5,
        init= "he_normal",
        activation='relu',
        subsample=(5, 4),
        border_mode='valid'), input_shape=input_shape))
    model.add(TimeDistributed(Convolution2D(32, 5, 5,
        init= "he_normal",
        activation='relu',
        subsample=(3, 2),
        border_mode='valid')))
    model.add(TimeDistributed(Convolution2D(48, 3, 3,
        init= "he_normal",
        activation='relu',
        subsample=(1,2),
        border_mode='valid')))
    model.add(TimeDistributed(Convolution2D(64, 3, 3,
        init= "he_normal",
        activation='relu',
        border_mode='valid')))
    model.add(TimeDistributed(Convolution2D(128, 3, 3,
        init= "he_normal",
        activation='relu',
        subsample=(1,2),
        border_mode='valid')))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(
        output_dim=256,
        init='he_normal',
        activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(
        2,
        init='he_normal',
        activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', 
                  metrics=['accuracy'])

    return model

def get_labels():
    with open('./inception/retrained_labels.txt', 'r') as fin:
        labels = [line.rstrip('\n') for line in fin]
    return labels

def _frame_generator(batch, batch_size, num_frames):
    """Generate batches of frames to train on. Batch for memory."""
    image_path = 'images/' + batch + '/'

    labels = get_labels()
    
    with open('data/labeled-frames-' + batch + '.pkl', 'rb') as fin:
        frames = pickle.load(fin)

    print(len(frames))
    print('------------------------')

    X = deque()
    batch_X = []
    batch_y = []
    for i, frame in enumerate(frames):
        # Get one-hot encoded y.
        label = frame[1]
        y = [int(x == label) for x in labels]

        # Get the image, which is a piece of X.
        filename = frame[0]
        image = image_path + filename + '.jpg'
        image_data = imread(image)

        # Re-order the indices for TF.
        image_data = image_data.transpose(2, 0, 1)

        # Add X to the dequeue and remove the old one, if we're full.
        X.append(image_data)
        if len(X) == num_frames + 1:
            X.popleft()

        # Add the data to our batch.
        batch_X.append(X)
        batch_y.append(y)
            
        # If our batch is full, yield it.
        if i > 0 and i % (batch_size - 1) == 0:
            # Yield then reset for the next batch.
            yield np.array(batch_X), np.array(batch_y)
            batch_X = []
            batch_y = []

def main():
    batch = '1'
    num_frames = 10
    batch_size = 32
    input_shape = (num_frames, 3, 240, 320)
    model = get_model(input_shape)
    model.fit_generator(
        _frame_generator(batch, batch_size, num_frames),
        samples_per_epoch=1000,
        nb_epoch=10
    )

if __name__ == '__main__':
    main()
