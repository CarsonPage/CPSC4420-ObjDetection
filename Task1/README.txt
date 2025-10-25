[*]--------------------------------------------[*]
 |            Object Detection with             |
 |           OpenCV + MobileNet-SSD!            |
 |           	                                |
 |	   CPSC 4420 Project: First Task        |
 |              For Hazim Alzorgan              |
 |                                              |
 |	 	   Project by:                  |
 |       Carson Page (cnpage@clemson.edu)       |
 |	Xander Facey (afacey@clemson.edu)       |
[*]--------------------------------------------[*]

For this task, we needed to deliver Python Code that accomplishes two things:
	1. Uses a webcam feed as input.
	2. Detects and highlights objects in real-time with bounding boxes.

For this complex of a task and such a short turn-around window, we elected to use a
pre-existing, open-source DNN (Deep Neural Network). After researching, we landed on
MobileNet-SSD. This is due to it being designed to run on mobile/embedded devices and 
its ability to function with limited computational power, perfect for our Raspberry Pi
use case. MobileNet-SSD was developed by Google and is widely used for creating real-
time object detection systems.

This is used in tandem with OpenCV, a Python library for real-time computer vision,
image processing, and machine learning tasks. This is necessary to process webcam
output for MobileNet-SSD to use for object detection.

[*]------------------[*]
 | Setup Instructions |
[*]------------------[*]

TO COMPILE YOURSELF:

	1. Install dependencies:
   	```bash
   	pip install opencv-python
   	```
	2. Download the pretrained MobileNet SSD model files:
   	- [deploy.prototxt](https://github.com/chuanqi305/MobileNet-SSD/blob/master/deploy.prototxt)
   	- [mobilenet_iter_73000.caffemodel](https://github.com/chuanqi305/MobileNet-SSD/blob/master/mobilenet_iter_73000.caffemodel)

   	Place both files in the same directory as `object_detection.py`.

	3. Run the program:
	```bash
	python object_detection.py
   	```

	4. A window will open showing the live webcam feed with bounding boxes.
   			     ---Press **ESC** to exit.---

ALTERNATIVELY, you can run "object_detection.exe", which we made using PyInstaller:
	The application bundles the dependencies into one file, allowing you to click and run without the hassle!


[*]-----------------[*]
 |       Notes       |
[*]-----------------[*]

- You can adjust the confidence threshold in the code to control sensitivity.
- The .exe version does not need to be in the same folder as its dependencies in order to run. The dependencies are packaged with the executable.
