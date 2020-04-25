<h2>Convolutional Neural Network Object Detection YOLOv2</h2>

Depdencies:
OpenCV, imutils, numpy

YOLO Object Detection is based on the darknet project https://github.com/pjreddie/darknet

*The issue is the threshold value for detection here is fixed. 
*Lighting in the camera image effects the threshold value (when video is dark, the threshold value is lower and when video is light, the threshold value must be set higher) 

Added

- sound plays when object detection threshold value is passed 
- transparent solid boxes over detection plane
