from ultralytics import YOLO
from PIL import Image
import cv2
import os
import pyk4a
from pyk4a import Config, PyK4A
import numpy as np
import time

k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=True,
        )
    )
k4a.start()

model = YOLO("yolov8x-cls.pt")

# getters and setters directly get and set on device
k4a.whitebalance = 4500
assert k4a.whitebalance == 4500
k4a.whitebalance = 4510
assert k4a.whitebalance == 4510

start_time = time.time()
flag = True

while 1:
    capture = k4a.get_capture()
    if np.any(capture.color):
        color_data_rgb = capture.color[:, :, :3]

        height, width = 500, 320
        color_data_rgb_trimmed = color_data_rgb[:height, :width, :]

        cv2.imshow("k4a", color_data_rgb_trimmed)


        current_time = time.time()

        # # We will print an output on terminal after every five seconds
        # if current_time - start_time >= 10 and flag:
        #     results = model.predict(source=color_data_rgb_trimmed)

        #     # Find the index with the highest probability
        #     best_class_idx = results[0].probs.argmax()

        #     # Get the class name with the highest probability
        #     class_names = list(results[0].names.values())
        #     best_class_name = class_names[best_class_idx]

        #     print(f'this is a {best_class_name}')

        #     # Reset the timings of the start time
        #     start_time = current_time
        #     flag = False

        key = cv2.waitKey(10)
        if key != -1:
            cv2.destroyAllWindows()
            break
k4a.stop()
