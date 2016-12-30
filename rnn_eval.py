"""
Run a holdout set of data through our trained RNN. Requires we first
run train_rnn.py and save the weights.
"""
from rnn_utils import get_network, get_network_deep, get_network_wide, get_data
import tflearn

def main(filename, frames, batch_size, num_classes, input_length):
    """From the blog post linked above."""
    # Get our data.
    X_train, _, y_train, _ = get_data(filename, frames, num_classes, input_length)

    # Get sizes.
    num_classes = len(y_train[0])

    # Get our network.
    net = get_network_wide(frames, input_length, num_classes)

    # Get our model.
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.load('checkpoints/rnn.tflearn')

    # Evaluate.
    print(model.evaluate(X_train, y_train))

if __name__ == '__main__':
    filename = 'data/predicted-frames-2.pkl'
    input_length = 2
    # filename = 'data/cnn-features-frames-2.pkl'
    # input_length = 2048
    frames = 40
    batch_size = 32
    num_classes = 2

    main(filename, frames, batch_size, num_classes, input_length)
