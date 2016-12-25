"""
Class to read/stream images off the PiCamera.
"""
import time
from picamera import PiCamera

def capture_images(save_folder):
    """Stream images off the camera and save them."""
    camera = PiCamera()
    camera.resolution = (320, 240)
    camera.framerate = 5

    # Warmup...
    time.sleep(2)

    # And capture continuously forever.
    for _ in camera.capture_continuous(
            save_folder + '{timestamp}.jpg',
            'jpeg', use_video_port=True
    ):
        pass

if __name__ == '__main__':
    capture_images('/mnt/usbstorage/images/')
