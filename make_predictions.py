"""
Classify all the images in a holdout set and score.
"""
import pickle
import sys
import tensorflow as tf
from tqdm import tqdm

def get_labels():
    """Return a list of our trained labels so we can
    test our training accuracy. The file is in the
    format of one label per line, in the same order
    as the predictions are made. The order can change
    between training runs."""
    with open("./inception/retrained_labels.txt", 'r') as fin:
        labels = [line.rstrip('\n') for line in fin]
    return labels

def predict_on_frames(frames, batch):
    """Given a list of frames, predict all their classes."""
    # Unpersists graph from file
    with tf.gfile.FastGFile("./inception/retrained_graph.pb", 'rb') as fin:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fin.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        frame_predictions = []
        image_path = 'images/' + batch + '/'
        pbar = tqdm(total=len(frames))
        for i, frame in enumerate(frames):
            filename = frame[0]
            label = frame[1]

            # Get the image path.
            image = image_path + filename + '.jpg'

            # Read in the image_data
            image_data = tf.gfile.FastGFile(image, 'rb').read()

            try:
                predictions = sess.run(
                    softmax_tensor,
                    {'DecodeJpeg/contents:0': image_data}
                )
                prediction = predictions[0]
            except KeyboardInterrupt:
                print("You quit with ctrl+c")
                sys.exit()
            except:
                print("Error making prediction, continuing.")
                continue

            # Save the probability that it's each of our classes.
            frame_predictions.append([prediction, label])

            if i > 0 and i % 10 == 0:
                pbar.update(10)

        pbar.close()

        return frame_predictions

def get_accuracy(predictions, labels):
    """After predicting on each batch, check that batch's
    accuracy to make sure things are good to go. This is
    a simple accuracy metric, and so doesn't take confidence
    into account, which would be a better metric to use to
    compare changes in the model."""
    correct = 0
    for frame in predictions:
        # Get the highest confidence class.
        this_prediction = frame[0].tolist()
        this_label = frame[1]

        max_value = max(this_prediction)
        max_index = this_prediction.index(max_value)
        predicted_label = labels[max_index]

        # Now see if it matches.
        if predicted_label == this_label:
            correct += 1

    accuracy = correct / len(predictions)
    return accuracy

def main():
    batches = ['1']
    labels = get_labels()

    for batch in batches:
        print("Doing batch %s" % batch)
        with open('data/labeled-frames-' + batch + '.pkl', 'rb') as fin:
            frames = pickle.load(fin)

        # Predict on this batch and get the accuracy.
        predictions = predict_on_frames(frames, batch)
        accuracy = get_accuracy(predictions, labels)
        print("Batch accuracy: %.5f" % accuracy)

        # Save it.
        with open('data/predicted-frames-' + batch + '.pkl', 'wb') as fout:
            pickle.dump(predictions, fout)

    print("Done.")

if __name__ == '__main__':
    main()
