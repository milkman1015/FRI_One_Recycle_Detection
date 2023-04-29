# FRI_One_Recycle_Detection

pip3 install ultralytics

pip3 install pyk4a

Description
This project uses YOLO-8 for object detection, OpenCV for video frame extraction, and ChatGPT for determining if the detected object is recyclable or not. The system utilizes Azure Kinect Video Stream as the video source for real-time object classification.

Dependencies
YOLO-8: Deep learning object detection model for image classification 
OpenCV: Open source computer vision library for video frame processing 
Azure Kinect SDK: Required for integrating the Azure Kinect Video Stream 
ChatGPT: OpenAI's GPT-4 based model for natural language processing 

Implementation
We used the YOLO-8 model for classifying objects in images. OpenCV was utilized to process the video stream, where frames were extracted and sent to YOLO-8 for object detection. The detected objects were then sent to ChatGPT, which employed the highest weighted index from the predicted results to determine the object's recyclability. The video source for the project was the Azure Kinect Video Stream, captured using OpenCV.
