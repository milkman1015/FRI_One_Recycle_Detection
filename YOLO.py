from ultralytics import YOLO
from PIL import Image
import cv2
import os
# import openai

model = YOLO("yolov8x-cls.pt")

# **************** YOLO ON AN IMAGE ****************
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
#results = model.predict(source="0")
#results = model.predict(source="folder", show=True) # Display preds. Accepts all YOLO predict arguments

# uncomment this block for image
"""
# from PIL
im1 = Image.open("/home/bwilab/group10_FRI_final/FRI_One_Recycle_Detection/testImages/attachments/water_bottle_img.jpg")
# results = model.predict(source=im1, save=False)  # save plotted images

# from list of PIL/ndarray
results = model.predict(source=im1)

# Find the index with the highest probability
best_class_idx = results[0].probs.argmax()

# Get the class name with the highest probability
class_names = list(results[0].names.values())
best_class_name = class_names[best_class_idx]

print(f'this is a {best_class_name}')

# print("class with highest confidence")
# print(best_class_name)

# print("all classes")
# print(class_names)
"""

# **************** YOLO ON A VIDEO ****************
# open a video
cap = cv2.VideoCapture('/home/bwilab/group10_FRI_final/FRI_One_Recycle_Detection/testImages/attachments/water_bottle_vid.MOV')

# retrieve the frame and whether or not it was successful in retrieval
ret, frame = cap.read()
counter = 1  # count which frame we are on

framerate = 30  # frames per second of the video being read
seconds = 3  # number of seconds between each reading

# while there are frames in the video
while ret:
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
    
    # retrieve next frame and bool
    ret, frame = cap.read()
    counter += 1  # increment counter

# release the video and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


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
"""

os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.getenv("OPENAI_API_KEY")
# all models down below
# openai.Model.list()

# Set the messages for the chat
messages = [
    {"role": "user", "content": f"Is the object '{best_class_name}' recyclable or not recyclable? Please provide a one-word answer."},
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
print(best_class_name + ": " + assistant_response)

"""

