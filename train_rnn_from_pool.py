"""
Given the output of our Inception classifier, can we train
a RNN to ultimately decide whether to mute or not?

The intuition here is that some sequence of ads/not ads
will ultimately mean mute. We could just say:
"If X frames are Y at confidence Z, do A". But why write
heuristics when a neural network can do it for you?

With love:
http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/
"""
from collections import deque
from sklearn.model_selection import train_test_split
import keras
import logger
import numpy as np
import pickle
import sys
import tensorflow as tf

def get_data(batches, num_frames, one_hot=True):
    # Get our files for training.
    logger.logging.info("Getting data.")

    # Local vars.
    X = []
    y = []

    for batch in batches:
        temp_list = deque()  # Reset it each batch.

        with open('data/cnn-features-frames-' + batch + '.pkl', 'rb') as fin:
            frames = pickle.load(fin)

            logger.logging.info("Found %d frames." % len(frames))

            # Depending on the batch, we have different frame rates.
            # Downsample them all to 1fps.
            if batch == '1' or batch == '2' or batch == '3' \
                    or batch == '5' or batch == '6' or batch == '7':
                skip_frames = 10
            else:
                skip_frames = 5

            for i, frame in enumerate(frames):
                features = frame[0]
                actual = frame[1]

                # Convert our many labels into binary.
                if actual == 'ad':
                    actual = 1
                else:
                    actual = 0

                # Skip frames to downsample.
                if i % skip_frames != 0:
                    continue

                # Reshape the features.
                features = features[0][0]
                features = features.ravel()

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

    logger.logging.info("Total dataset size: %d" % len(X))
    logger.logging.info("Total recording time: %.2f" % (len(X) / 60 / 60))

    # Numpy the y.
    y = np.array(y)

    # One-hot encoded categoricals.
    if one_hot:
        y = keras.utils.np_utils.to_categorical(y, 3)

    # Split into train and test.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42)

    return X_train, X_test, y_train, y_test

def main(batches, frames, batch_size, epoch):
    # Tensorflow stuff.
    # {frames} frames, 2D
    # 3 outputs
    data = tf.placeholder(tf.float32, [None, frames, 2048])
    target = tf.placeholder(tf.float32, [None, 3])

    # Create our layer.
    num_hidden = 24
    cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)

    # Unroll.
    val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

    # Transpose and gather.
    val = tf.transpose(val, [1, 0, 2])
    last = tf.gather(val, int(val.get_shape()[0]) - 1)

    # Thank goodness for tutorials...
    weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
    bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))

    # Prediction layer.
    prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)

    # Scoring layer.
    cross_entropy = -tf.reduce_sum(target * tf.log(prediction))

    # Now prepare the optimizer.
    optimizer = tf.train.AdamOptimizer()
    minimize = optimizer.minimize(cross_entropy)

    # To calculate the error on the test data.
    mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
    error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

    # Now that the model is built, let's execute it.
    init_op = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init_op)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Get our data.
    X_train, X_test, y_train, y_test = get_data(batches, frames)

    # Train it!
    no_of_batches = int(len(X_train)/batch_size)
    for i in range(epoch):
        ptr = 0
        for j in range(no_of_batches):
            inp, out = X_train[ptr:ptr+batch_size], y_train[ptr:ptr+batch_size]
            ptr += batch_size
            sess.run(minimize, {data: inp, target: out})
        incorrect = sess.run(error,{data: X_test, target: y_test})
        logger.logging.info('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
    saver.save(sess, 'checkpoints/rnn_checkpoint.ckpt')
    sess.close()

    # Return the last error rate.
    return incorrect

if __name__ == '__main__':
    batches = ['1', '2', '3', '5', '6', '7', '8', '9', '10', '11', '12']

    frames = 5
    batch_size = 32
    epochs = 30

    error = main(batches, frames, batch_size, epochs)
