from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO("yolov8x-cls.pt")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
# results = model.predict(source="0")
# results = model.predict(source="folder", show=True) # Display preds. Accepts all YOLO predict arguments

# from PIL
im1 = Image.open("/home/bwilab/Documents/glassBottle.png")
results = model.predict(source=im1, save=True)  # save plotted images


# from list of PIL/ndarray
results = model.predict(source=im1)
print(results[0])