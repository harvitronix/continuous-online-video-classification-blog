"""
Combining CNN and RNN into a CRNN, using Keras' TimeDistributed wrapper.
"""
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
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
    model.add(LSTM(64, dropout_W=0.2, dropout_U=0.2, return_sequences=True))
    model.add(LSTM(64, dropout_W=0.2, dropout_U=0.2, return_sequences=True))
    model.add(LSTM(64, dropout_W=0.2, dropout_U=0.2))
    model.add(Dropout(0.2))
    model.add(Dense(
        output_dim=256,
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
    
    with open('data/labeled-frames-' + batch + '.pkl', 'rb') as fin:
        frames = pickle.load(fin)

    print(len(frames))
    print('------------------------')

    X = deque()
    batch_X = []
    batch_y = []
    for i, frame in enumerate(frames):
        # Get y.
        label = frame[1]

        # Get the image, which is a piece of X.
        filename = frame[0]
        image = image_path + filename + '.jpg'
        image_data = imread(image)

        # Add X to the dequeue and remove the old one, if we're full.
        X.append(image_data)
        if len(X) == num_frames + 1:
            X.popleft()

        # Add the data to our batch.
        batch_X.append(X)
        batch_y.append(label)
            
        # If our batch is full, yield it.
        if i > 0 and i % (batch_size - 1) == 0:
            # First numpy them, then reset our batch lists.
            batch_X_np = np.array(batch_X)
            batch_y_np = np.array(batch_y)
            batch_X = []
            batch_y = []
            yield batch_X_np, batch_y_np

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
