from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO("yolov8x-cls.pt")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
# results = model.predict(source="0")
# results = model.predict(source="folder", show=True) # Display preds. Accepts all YOLO predict arguments

# from PIL
im1 = Image.open("/home/bwilab/Wyatt_Aman_Sahir_John_FRI1/FRI_One_Recycle_Detection/testImages/plasticBag.png")
results = model.predict(source=im1, save=False)  # save plotted images


# from list of PIL/ndarray
results = model.predict(source=im1)

# Find the index with the highest probability
best_class_idx = results[0].probs.argmax()

# Get the class name with the highest probability
class_names = list(results[0].names.values())
best_class_name = class_names[best_class_idx]

print(best_class_name)