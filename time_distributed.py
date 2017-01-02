"""
Combining CNN and RNN into a CRNN, using Keras' TimeDistributed wrapper.
"""
from keras.callbacks import TensorBoard
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from scipy.misc import imread
from collections import deque
import random
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
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', 
                  metrics=['accuracy'])

    return model

def get_labels():
    with open('./inception/retrained_labels.txt', 'r') as fin:
        labels = [line.rstrip('\n') for line in fin]
    return labels

def get_sequences(batch, num_frames, labels):
    """
    Create a list that just holds sequences of frames with their filenames.
    This is so we can shuffle and train without losing the order
    of frames within each batch. Should prevent over-fitting to a specific
    batch, where all classes for that batch are the same.

    So the result will be something like:
    [[filename1, filename2, ...], [filename33, filename34, ...]]
    """
    with open('data/labeled-frames-' + batch + '.pkl', 'rb') as fin:
        frames = pickle.load(fin)
        X = []
        y = []
        image_queue = deque()

        for i, frame in enumerate(frames):
            image = frame[0]
            label = frame[1]

            # One-hot encode our label.
            y_onehot = [int(x == label) for x in labels]

            if len(image_queue) == num_frames - 1:
                image_queue.append(image)
                X.append(list(image_queue))
                y.append(y_onehot)
                image_queue.popleft()
            else:
                image_queue.append(image)
                continue

        return X, y

def get_image_data(image, batch):
    """Given an image filename, return the numpy version of the image
    in the correct shape.
    """
    base_path = 'images/' + batch + '/'

    # Get the image data.
    image_path = base_path + image + '.jpg'
    image_data = imread(image_path)
                
    return image_data

def _frame_generator(batch, batch_size, num_frames):
    """Generate batches of frames to train on. Batch for memory."""
    labels = get_labels()  # Call here to do it once.
    sequences, y_true = get_sequences(batch, num_frames, labels)
    
    while 1:
        # Get a random sample equal to our batch_size.
        random_ints = random.sample(range(0, len(sequences) - 1), batch_size)
        samples = [sequences[x] for x in random_ints]
        ys = [y_true[x] for x in random_ints]

        batch_X = []
        batch_y = []

        # Loop through each sample.
        for i, sample in enumerate(samples):
            sequence = [get_image_data(x, batch) for x in sample]

            batch_X.append(sequence)
            batch_y.append(ys[i])

        # Now yield it.
        yield np.array(batch_X), np.array(batch_y)

def train():
    print('*****Training.*****')
    batch = '1'
    num_frames = 10
    batch_size = 32
    input_shape = (num_frames, 240, 320, 3)
    tb = TensorBoard(log_dir='./logs')
    model = get_model(input_shape)
    model.fit_generator(
        _frame_generator(batch, batch_size, num_frames),
        samples_per_epoch=384,
        nb_epoch=50,
        validation_data=_frame_generator(batch, batch_size, num_frames),
        nb_val_samples=100,
        callbacks=[tb]
    )
    model.save('checkpoints/crnn.h5')

def evaluate():
    print('*****Evaluating.*****')
    batch = '2'
    num_frames = 10
    batch_size = 32
    input_shape = (num_frames, 240, 320, 3)
    tb = TensorBoard(log_dir='./logs')
    model = get_model(input_shape)
    model.load_weights('checkpoints/crnn.h5')
    score = model.evaluate_generator(
        _frame_generator(batch, batch_size, num_frames),
        val_samples=batch_size*10
    )
    print(score)
    print(model.metrics_names)

def main():
    train()
    evaluate()

if __name__ == '__main__':
    main()
