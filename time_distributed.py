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

def get_sequences(batches, num_frames, labels):
    """
    Create a list that just holds sequences of frames with their filenames.
    This is so we can shuffle and train without losing the order
    of frames within each batch. Should prevent over-fitting to a specific
    batch, where all classes for that batch are the same.

    So the result will be something like:
    [[filename1, filename2, ...], [filename33, filename34, ...]]
    """
    X_all = []
    y_all = []
    for batch in batches:
        base_path = 'images/' + batch + '/'
        X = []
        y = []

        with open('data/labeled-frames-' + batch + '.pkl', 'rb') as fin:
            frames = pickle.load(fin)

        print("Loading batch %s with %d frames." % (batch, len(frames)))
        image_queue = deque()

        for i, frame in enumerate(frames):
            image = frame[0]
            label = frame[1]

            # One-hot encode our label.
            y_onehot = [int(x == label) for x in labels]

            # Append to queue, add queue to X, pop from queue.
            image_path = base_path + image + '.jpg'
            if len(image_queue) == num_frames - 1:
                image_queue.append(image_path)
                X.append(list(image_queue))
                y.append(y_onehot)
                image_queue.popleft()
            else:
                image_queue.append(image_path)
                continue

        X_all += X
        y_all += y

    print("Return %d sequences." % len(X_all))
    return X_all, y_all

def get_image_data(image):
    """Given an image filename, return the numpy version of the image
    in the correct shape.
    """
    try:
        image_data = imread(image)
    except OSError:
        print("Couldn't find file %s" % image)
        return None
                
    return image_data

def _frame_generator(sequences, y_true, batch_size, num_frames):
    """Generate batches of frames to train on. Batch for memory."""
    while 1:
        # Get a random sample equal to our batch_size.
        random_ints = random.sample(range(0, len(sequences) - 1), batch_size)
        samples = [sequences[x] for x in random_ints]

        ys = [y_true[x] for x in random_ints]

        batch_X = []
        batch_y = []
        skip_batch = False

        # Loop through each sample.
        for i, sample in enumerate(samples):
            sequence = [get_image_data(x) for x in sample]

            # Make sure we don't have nones in there.
            if None in sequence:
                skip_batch = True
                break

            batch_X.append(sequence)
            batch_y.append(ys[i])

        if skip_batch:
            print("Skipping batch.")
            continue

        # Now yield it.
        yield np.array(batch_X), np.array(batch_y)

def train():
    print('*****Training.*****')
    # Set defaults.
    batches = ['1', '3', '5']
    num_frames = 10
    batch_size = 32
    input_shape = (num_frames, 240, 320, 3)
    tb = TensorBoard(log_dir='./logs')

    # Get labels and sequences once.
    labels = get_labels()
    X_train, y_train = get_sequences(batches, num_frames, labels)
    X_test, y_test = get_sequences(['2'], num_frames, labels)

    # Get the model.
    model = get_model(input_shape)
    print(model.summary())

    # Load weights from previous runs.
    # model.load_weights('checkpoints/crnn.h5')

    # Fit on batches passed by our generator. Validate on another batch.
    model.fit_generator(
        _frame_generator(X_train, y_train, batch_size, num_frames),
        samples_per_epoch=int(len(X_train) / batch_size),
        nb_epoch=10,
        validation_data=_frame_generator(X_test, y_test, batch_size, num_frames),
        nb_val_samples=100,
        callbacks=[tb]
    )
    model.save('checkpoints/crnn-take2.h5')

def evaluate():
    print('*****Evaluating.*****')
    # Set defaults.
    batches = ['1', '3', '5']
    num_frames = 10
    batch_size = 32
    input_shape = (num_frames, 240, 320, 3)

    labels = get_labels()
    X_val, y_val = get_sequences(['2'], num_frames, labels)

    model = get_model(input_shape)
    model.load_weights('checkpoints/crnn-take2.h5')
    score = model.evaluate_generator(
        _frame_generator(X_val, y_val, batch_size, num_frames),
        val_samples=batch_size*10
    )
    print(score)
    print(model.metrics_names)

def main():
    # train()
    evaluate()

if __name__ == '__main__':
    main()
