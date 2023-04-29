from ultralytics import YOLO
from PIL import Image
import cv2
import os
import pyk4a
from pyk4a import Config, PyK4A
import numpy as np
import time
import openai
import os
import glob

model = YOLO("yolov8x-cls.pt")

os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.getenv("OPENAI_API_KEY")


def classify_image(path):
    im1 = Image.open(path)
    results = model.predict(source=im1)
    # Find the index with the highest probability
    best_class_idx = results[0].probs.argmax()

    # Get the class name with the highest probability
    class_names = list(results[0].names.values())
    best_class_name = class_names[best_class_idx]

    print(f'this is a {best_class_name}')
    print(path)
    return best_class_name


def ask_gpt(class_name):
    messages = [
        {"role": "user", "content": f"Is the object '{class_name}' recyclable or not recyclable? Please provide a one-word answer."},
    ]

    # Make a request to the OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
    )

    # Extract the assistant's response from the API response
    assistant_response = response.choices[0].message.content.strip()

    # Print the assistant's response
    print(class_name + ": " + assistant_response)

    return assistant_response


def get_image_paths(folder_path):
    # Append image files' paths into the list and return it
    image_paths = []

    files = glob.glob(os.path.join(folder_path, '*'))

    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(file)

    return image_paths


def run_test():
    total = 0
    correct = 0
    incorrect = []

    # Runs tests on non recyclable items
    path = '/home/bwilab/FRI_FinalProject/testImages/Garbage'
    paths = get_image_paths(path)

    for file in paths:
        # Feed the path into YOLO, classify its image and
        # print the output.

        class_name = classify_image(file)

        response = ask_gpt(class_name)
        total += 1
        if response == 'Not recyclable.' or response == 'Not recyclable':
            correct += 1
        else:
            incorrect.append(class_name)

    # Runs tests on recyclable items

    path = '/home/bwilab/FRI_FinalProject/testImages/Recyclable'
    paths = get_image_paths(path)

    for file in paths:
        class_name = classify_image(file)

        response = ask_gpt(class_name)
        total += 1
        if response == 'Recyclable.' or response == 'Recyclable':
            correct += 1
        else:
            incorrect.append(class_name)

    # Prints out the data
    # 1. Total number of correctly identified items
    # 2. Prints out the class names it incorrectly classified.
    print(f'total: {total}, correct: {correct}\nincorrect classes: ', end=' ')
    for name in incorrect:
        print(name, ' ', end=' ')

    
def run_kinect():
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=True,
        )
    )

    # # Open the sensor with default configuration
    k4a.open()

    # Configure the color camera
    config = Config(color_resolution=pyk4a.ColorResolution.RES_1080P)

    # Start the color camera
    k4a.start()

    # getters and setters directly get and set on device
    k4a.whitebalance = 4500
    assert k4a.whitebalance == 4500
    k4a.whitebalance = 4510
    assert k4a.whitebalance == 4510

    start_time = time.time()
    # flag = True

    while 1:
        capture = k4a.get_capture()
        if np.any(capture.color):
            color_data_rgb = capture.color[:, :, :3]

            height, width = 500, 320
            color_data_rgb_trimmed = color_data_rgb[:height, :width, :]

            cv2.imshow("k4a", color_data_rgb_trimmed)


            current_time = time.time()

            # # We will print an output on terminal after every five seconds
            if current_time - start_time >= 10:
                results = model.predict(source=color_data_rgb_trimmed)

                # Find the index with the highest probability
                best_class_idx = results[0].probs.argmax()

                # Get the class name with the highest probability
                class_names = list(results[0].names.values())
                best_class_name = class_names[best_class_idx]

                print(f'this is a {best_class_name}')

                # Reset the timings of the start time
                start_time = current_time
                # flag = False

            key = cv2.waitKey(10)
            if key != -1:
                cv2.destroyAllWindows()
                break
    k4a.stop()


run_test()
run_kinect()
