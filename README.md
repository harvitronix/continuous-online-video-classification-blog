Step 1:
Capture a bunch of video.

Step 2:
Move each frame into its class directory in images/classifications/[classname]/ - you can do this automatically by defining the start and stop timestamps of each commercial in the commercials.py script, and then running build_labels.py. Be sure to set copyimage to True if you want it to copy the image. This also creates a reference file that we use later that identifies the class of each image.

Step 3:
Run the tensorflow script from the examples folder. Alternatively, we may want to edit and bring that script over to this repo.

Step 4:
Run build_predictions.py on the holdout set to see how it does.

Step 5:
Run the online system, classifying each frame with our newly trained weights.

