# Continuous online video classifcation

This repo holds the code that supports our blog posts about using TensorFlow, Inception and a Raspberry Pi for continuous online video classification.

Part 1 - CNN, Inception only: https://medium.com/@harvitronix/continuous-online-video-classification-with-tensorflow-inception-and-a-raspberry-pi-785c8b1e13e1

Part 2 - Adding an RNN (LSTM): https://medium.com/@harvitronix/continuous-video-classification-with-tensorflow-inception-and-recurrent-nets-250ba9ff6b85#.3vl3apzb6

## Overview

Step 1:

Capture a bunch of video with `stream_images.py`.

Step 2:

Move each frame into its class directory in images/classifications/[classname]/ - you can do this automatically by defining the start and stop timestamps of each commercial in the `commercials.py` script, and then running `build_labels.py`. Be sure to set copyimage to True if you want it to copy the image. This also creates a reference file that we use later that identifies the class of each image.

Step 3:

Run the `tensorflow/examples/image_retraining/retrain.py` script in the main TensorFlow repo. The full command we use is in the blog post linked above.

Step 4:

Run `make_predictions.py` on the holdout set to see how it does.

Step 5:

Run the online system with `online.py` on your Raspberry Pi, which will classify each frame captured with our newly trained weights.

