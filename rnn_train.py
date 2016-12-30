"""
Given a saved output of predictions or pooled features from our CNN,
train an RNN (LSTM) to examine temporal dependencies.
"""
from collections import deque
from rnn import get_network
from sklearn.model_selection import train_test_split
from tflearn.data_utils import to_categorical
from random import shuffle
import tflearn
import numpy as np
import pickle
import sys

def get_data(filename, num_frames, num_classes, input_length):
    """Get the data from our saved predictions or pooled features."""

    # Local vars.
    X = []
    y = []
    temp_list = deque()

    # Open and get the features.
    with open(filename, 'rb') as fin:
        frames = pickle.load(fin)

        for i, frame in enumerate(frames):
            features = frame[0]
            actual = frame[1]

            # Convert our labels into binary.
            if actual == 'ad':
                actual = 1
            else:
                actual = 0

            # Add to the queue.
            if len(temp_list) == num_frames - 1:
                temp_list.append(features)
                flat = list(temp_list)
                X.append(np.array(flat))
                y.append(actual)
                temp_list.popleft()
            else:
                temp_list.append(features)
                continue

    print("Total dataset size: %d" % len(X))

    # Numpy.
    X = np.array(X)
    y = np.array(y)

    # Reshape.
    X = X.reshape(-1, num_frames, input_length)

    # One-hot encoded categoricals.
    y = to_categorical(y, num_classes)

    # Split into train and test.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42)

    return X_train, X_test, y_train, y_test

def main(filename, frames, batch_size, num_classes, input_length):
    """From the blog post linked above."""
    # Get our data.
    X_train, X_test, y_train, y_test = get_data(filename, frames, num_classes, input_length)

    # Get sizes.
    num_classes = len(y_train[0])

    # Get our network.
    net = get_network(frames, input_length, num_classes)

    # Train the model.
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(X_train, y_train, validation_set=(X_test, y_test),
              show_metric=True, batch_size=batch_size, snapshot_step=100)

    # Save it.
    model.save('rnn.tflearn')

if __name__ == '__main__':
    filename = 'data/cnn-features-frames-1.pkl'
    frames = 40
    batch_size = 32
    num_classes = 2
    input_length = 2048

    main(filename, frames, batch_size, num_classes, input_length)
