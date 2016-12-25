# Continuous online video classifcation

This repo holds the code that supports our blog post about using TensorFlow, Inception and a Raspberry Pi for continuous online video classification.

The blog post is here:
TK

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

