import sys, os, cv2, shutil, tempfile

#Creates log.txt for crash reports/prints
sys.stdout = open("log.txt", "w")
sys.stderr = sys.stdout

####################################

#[*]----------------------------[*]#
# |  GET PATHS TO BUNDLED FILES  | #
#[*]----------------------------[*]#

#This block of code resolves any issues with finding the
#absolute path to this code's dependencies. This ensures
#that files can be found, whether this code is bundled
#into a .exe or ran independently as a script.

#Helper function that gets the correct absolute path:
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

#Get paths to bundled files:
prototxt_path = resource_path("deploy.prototxt")
caffemodel_path = resource_path("mobilenet_iter_73000.caffemodel")

#As a workaround, if PyInstaller can't resolve paths using
#resource_path(), then it will copy files to a safe
#temporary folder to ensure OpenCV still has a path to
#read from. This prevents the application from crashing
#when compiled as a .exe:
tmp_dir = tempfile.mkdtemp()
prototxt_path_safe = os.path.join(tmp_dir, "deploy.prototxt")
caffemodel_path_safe = os.path.join(tmp_dir, "mobilenet_iter_73000.caffemodel")

shutil.copy(prototxt_path, prototxt_path_safe)
shutil.copy(caffemodel_path, caffemodel_path_safe)

#Load network safely:
net = cv2.dnn.readNetFromCaffe(prototxt_path_safe, caffemodel_path_safe)

#Existence check for original files (for debugging)
if not os.path.exists(prototxt_path):
    raise FileNotFoundError(f"deploy.prototxt not found at {prototxt_path}")
if not os.path.exists(caffemodel_path):
    raise FileNotFoundError(f"caffemodel not found at {caffemodel_path}")

####################################

#[*]----------------------------[*]#
# |        IMPLEMENTATION        | #
#[*]----------------------------[*]#

#This block of code is the main loop of the program.
#It starts by opening the computer's webcam, then continuiously
#grabs frames and sends each frame through MobileNet-SSD's
#object detection network. It then draws green boxes around
#objects that are detected with >50% confidence. #This will be
#displayed as live-video with detection in an application window.
#The program will stop when the ESC key is pressed, and then the
#resources used are cleaned up.

#Start webcam
cap = cv2.VideoCapture(0)

#Print debug instructions to log.txt
print("[INFO] Starting live webcam feed... Press 'ESC' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    # Prepare the image for the network
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Loop over detections and draw bounding boxes
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the output
    cv2.imshow("Live Object Detection (MobileNet-SSD)", frame)

    # Exit on ESC key
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

####################################