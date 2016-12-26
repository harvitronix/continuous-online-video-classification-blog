"""
Find predictions that are wrong.
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

def predict_on_frames(frames, batch, labels):
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

            this_prediction = prediction.tolist()

            max_value = max(this_prediction)
            max_index = this_prediction.index(max_value)
            predicted_label = labels[max_index]

            if predicted_label != label:
                print(image_path + filename + '.jpg')
                print(predicted_label, label)
                print(prediction)
                print('------')

            if i > 0 and i % 10 == 0:
                pbar.update(10)

        pbar.close()

        return frame_predictions

def main():
    batches = ['2']
    labels = get_labels()

    for batch in batches:
        print("Doing batch %s" % batch)
        with open('data/labeled-frames-' + batch + '.pkl', 'rb') as fin:
            frames = pickle.load(fin)

        # Predict on this batch and get the accuracy.
        predictions = predict_on_frames(frames, batch, labels)

    print("Done.")

if __name__ == '__main__':
    main()
