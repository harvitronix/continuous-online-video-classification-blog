"""
Given a saved output of predictions or pooled features from our CNN,
train an RNN (LSTM) to examine temporal dependencies.
"""
from rnn_utils import get_network, get_network_deep, get_network_wide, get_data
import tflearn

def main(filename, frames, batch_size, num_classes, input_length):
    """From the blog post linked above."""
    # Get our data.
    X_train, X_test, y_train, y_test = get_data(filename, frames, num_classes, input_length)

    # Get sizes.
    num_classes = len(y_train[0])

    # Get our network.
    net = get_network_wide(frames, input_length, num_classes)

    # Train the model.
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(X_train, y_train, validation_set=(X_test, y_test),
              show_metric=True, batch_size=batch_size, snapshot_step=100,
              n_epoch=4)

    # Save it.
    model.save('checkpoints/rnn.tflearn')

if __name__ == '__main__':
    # filename = 'data/cnn-features-frames-1.pkl'
    # input_length = 2048
    filename = 'data/predicted-frames-1.pkl'
    input_length = 2
    frames = 40
    batch_size = 32
    num_classes = 2

    main(filename, frames, batch_size, num_classes, input_length)
