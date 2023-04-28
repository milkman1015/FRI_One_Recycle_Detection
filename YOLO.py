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



# not needed, can erase, may come in handy for compiling test image folder
# identified recyable classes below
# recyclable_classes = [
# 'beer_bottle','beer_glass','binder','bottlecap','can', 'can_opener''carton','CD_player',
# 'cellular_telephone','computer_keyboard','desktop_computer','digital_clock','digital_watch',
# 'dishwasher','electric_fan','electric_locomotive','envelope','hard_disc','iPod','laptop',
# 'loudspeaker','magazine','microwave','newspaper','plastic_bag','printer','remote_control',
# 'rubber_eraser','soda_bottle','stapler','television','toaster','trash_can','vacuum',
# 'washing_machine'
# ]

# not needed, can erase, may come in handy for compiling test images folder
# identified most non-recyable classes below
# non_recyclable_classes = [
# 'tench', 'goldfish', 'great_white_shark', 'tiger_shark', 'hammerhead', 'electric_ray', 
# 'stingray', 'cock', 'hen', 'ostrich', 'brambling', 'goldfinch', 'house_finch', 'junco', 
# 'indigo_bunting', 'robin', 'bulbul', 'jay', 'magpie', 'chickadee', 'water_ouzel', 'kite', 
# 'bald_eagle', 'vulture', 'great_grey_owl', 'European_fire_salamander', 'common_newt', 'eft',
# 'spotted_salamander', 'axolotl', 'bullfrog', 'tree_frog', 'tailed_frog', 'loggerhead', 
# 'leatherback_turtle', 'mud_turtle', 'terrapin', 'box_turtle', 'banded_gecko', 'common_iguana', 
# 'American_chameleon', 'whiptail', 'agama', 'frilled_lizard', 'alligator_lizard', 'Gila_monster', 
# 'green_lizard', 'African_chameleon', 'Komodo_dragon', 'African_crocodile', 'American_alligator', 
# 'triceratops', 'thunder_snake', 'ringneck_snake', 'hognose_snake', 'green_snake', 'king_snake', 
# 'garter_snake', 'water_snake', 'vine_snake', 'night_snake', 'boa_constrictor', 'rock_python', 
# 'Indian_cobra', 'green_mamba', 'sea_snake', 'horned_viper', 'diamondback', 'sidewinder', 
# 'trilobite', 'harvestman', 'scorpion', 'black_and_gold_garden_spider', 'barn_spider', 
# 'garden_spider', 'black_widow', 'tarantula', 'wolf_spider', 'tick', 'centipede', 'black_grouse', 
# 'ptarmigan', 'ruffed_grouse', 'prairie_chicken', 'peacock', 'quail', 'partridge', 'African_grey', 
# 'macaw', 'sulphur-crested_cockatoo', 'lorikeet', 'coucal', 'bee_eater', 'hornbill', 'hummingbird', 
# 'jacamar', 'toucan', 'drake', 'red-breasted_merganser', 'goose', 'black_swan', 'tusker', 
# 'echidna', 'platypus', 'wallaby', 'koala', 'wombat', 'jellyfish', 'sea_anemone', 'brain_coral', 
# 'flatworm', 'nematode', 'conch', 'snail', 'slug', 'sea_slug', 'chiton', 'chambered_nautilus', 
# 'Dungeness_crab', 'rock_crab', 'fiddler_crab', 'king_crab', 'American_lobster', 'spiny_lobster', 
# 'crayfish', 'hermit_crab', 'isopod', 'white_stork', 'black_stork', 'spoonbill', 'flamingo', 
# 'little_blue_heron', 'American_egret', 'bittern', 'crane_(bird)', 'limpkin', 'European_gallinule', 
# 'American_coot', 'bustard', 'ruddy_turnstone', 'red-backed_sandpiper', 'redshank', 'dowitcher', 
# 'aystercatcher', 'hand_blower', 'hand-held_computer', 'handkerchief', 'hard_disc', 'harmonica', 
# 'harp', 'harvester', 'hatchet', 'holster', 'home_theater', 'honeycomb', 'hook', 'hoopskirt', 
# 'horizontal_bar', 'horse_cart', 'hourglass', 'iPod', 'iron', 'jack-o-lantern', 'jean', 'jeep', 
# 'jersey', 'jigsaw_puzzle', 'jinrikisha', 'joystick', 'kimono', 'knee_pad', 'knot', 'lab_coat', 
# 'ladle', 'lampshade', 'laptop', 'lawn_mower', 'lens_cap', 'letter_opener', 'library', 'lifeboat', 
# 'lighter', 'limousine', 'liner', 'lipstick', 'Loafer', 'lotion', 'loudspeaker', 'loupe', 
# 'lumbermill', 'magnetic_compass', 'mailbag', 'mailbox', 'maillot', 'maillot', 'manhole_cover', 
# 'maraca', 'marimba', 'mask', 'matchstick', 'maypole', 'maze', 'measuring_cup', 'medicine_chest', 
# 'megalith', 'microphone', 'microwave', 'military_uniform', 'milk_can', 'minibus', 'miniskirt', 
# 'minivan', 'missile', 'mitten', 'mixing_bowl', 'mobile_home', 'Model_T', 'modem', 'monastery', 
# 'monitor', 'moped', 'mortar', 'mortarboard', 'mosque', 'mosquito_net', 'motor_scooter', 
# 'mountain_bike', 'mountain_tent', 'mouse', 'mousetrap', 'moving_van', 'muzzle', 'nail', 'neck_brace', 
# 'necklace', 'nipple', 'notebook', 'obelisk', 'oboe', 'ocarina', 'odometer', 'oil_filter', 'organ', 
# 'oscilloscope', 'overskirt', 'oxcart', 'o_oxygen_mask', 'packet', 'paddle', 'paddlewheel', 'padlock', 
# 'paintbrush', 'pajama', 'palace', 'panpipe', 'paper_towel', 'parachute', 'parallel_bars', 'park_bench', 
# 'parking_meter', 'passenger_car', 'patio', 'pay-phone', 'pedestal', 'pencil_box', 'pencil_sharpener', 'perfume', 
# 'Petri_dish', 'photocopier', 'pick', 'pickelhaube', 'picket_fence', 'pickup', 'pier', 'piggy_bank', 'pill_bottle', 
# 'pillow', 'ping-pong_ball', 'pinwheel', 'pirate', 'pitcher', 'plane', 'planetarium', 'plastic_bag', 'plate_rack', 
# 'plow', 'plunger', 'Polaroid_camera', 'pole', 'police_van', 'poncho', 'pool_table', 'pop_bottle', 'pot', 'potter', 
# 's_wheel', 'power_drill', 'prayer_rug', 'printer', 'prison', 'projectile', 'projector', 'puck', 'punching_bag', 
# 'purse', 'quill', 'quilt', 'racer', 'racket', 'radiator', 'radio', 'radio_telescope', 'rain_barrel', 
# 'recreational_vehicle', 'reel', 'reflex_camera'
# ]
 
# gpt API, not needed currently so commented as to not waste money



# **************** YOLO ON A VIDEO ****************
# Initialize the Azure Kinect sensor
'''
k4a = PyK4A(
    Config(
        color_resolution=pyk4a.ColorResolution.RES_720P,
        depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
        synchronized_images_only=True,
    )
)

# # Open the sensor with default configuration
# k4a.open()

# # Configure the color camera
# config = Config(color_resolution=pyk4a.ColorResolution.RES_1080P)

# Start the color camera
k4a.start()

# # getters and setters directly get and set on device
# k4a.whitebalance = 4500
# assert k4a.whitebalance == 4500
# k4a.whitebalance = 4510
# assert k4a.whitebalance == 4510

while True:
    capture = k4a.get_capture()
    color_frame = capture.color

    color_image = color_frame[:, :, :3].copy()
    # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    
    # Display the color frame
    cv2.imshow('Color Frame', color_image)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop the sensor and close the window
k4a.stop()
cv2.destroyAllWindows()
'''

'''
# open a video
cap = cv2.VideoCapture('/home/bwilab/FRI_FinalProject/testImages/attachments/water_bottle_vid.MOV')

# retrieve the frame and whether or not it was successful in retrieval
ret, frame = cap.read()
counter = 1  # count which frame we are on

framerate = 30  # frames per second of the video being read
seconds = 1  # number of seconds between each reading

# while there are frames in the video
while ret:
    # from list of PIL/ndarray
    # results = model.predict(source=color_frame)

    # # Find the index with the highest probability
    # best_class_idx = results[0].probs.argmax()

    # # Get the class name with the highest probability
    # class_names = list(results[0].names.values())
    # best_class_name = class_names[best_class_idx]

    # print(f'this is a {best_class_name}')

    # capture = k4a.get_capture()
    # color_frame = capture.color

    # only read when we want too so we dont take too long
    if counter == framerate * seconds:
        # from list of PIL/ndarray
        results = model.predict(source=frame)

        # Find the index with the highest probability
        best_class_idx = results[0].probs.argmax()

        # Get the class name with the highest probability
        class_names = list(results[0].names.values())
        best_class_name = class_names[best_class_idx]
    
        # print the class name with the highest probability
        print(f'\nThe class with the highest confidence right now is: {best_class_name}\n')
        counter = 0  # reset the frame counter
    
    # # retrieve next frame and bool
    ret, frame = cap.read()
    counter += 1  # increment counter

# release the video and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
'''

