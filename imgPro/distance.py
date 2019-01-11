from imutils import paths
import numpy as np
import imutils
import cv2

def find_marker(image):
    #conv to grayscale, blur it, detect edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(gray, 35, 125)

    cv2.imshow("test",edged)
    cv2.waitKey(0)

    #find the contours in the edged image, keep the largest one; we'll asume that this is our object in the emage

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    #compute the bounding box of the object region and return it

    return cv2.minAreaRect(c)


def distance_to_camera(knownWidth, focalLength, perWidth):
    return (knownWidth*focalLength) / perWidth


#initialize the known distance from the camera to the object (cm)
KNOWN_DISTANCE = 50.0

#initialize the known object width
KNOWN_WIDTH = 21.0

#load the image with an object known to be 50 cm from the camera then find marker in image and initialize focal length
image = cv2.imread("img/distanceTest/TESTaopencv0.jpg")
marker = find_marker(image)
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

camera = cv2.VideoCapture(0)
if camera.isOpened():
    while (True):
        # cv2.waitKey(0)
        value, image = camera.read()
        cv2.imwrite("img/distanceTest/aopencv" + str(0) + '.jpg', image)
        cv2.imshow("image", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            camera.release()
            break
else:
    print("No camera")


del(camera)


"""for imagePath in sorted(paths.list_images("img/distanceTest")):
    #load the image, find the marker in the image then compute the distance from the marker to the camera
    image = cv2.imread(imagePath)
    marker = find_marker(image)
    cm = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])

    #draw a bounding box around the image and display it
    box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
    box = np.int0(box)
    cv2.drawContours(image, [box], -1, (0,255,0), 2)
    cv2.putText(image, "%.2fcm" % (cm), (image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    break
"""
imagePath = "img/distanceTest/aopencv0.jpg"

image = cv2.imread(imagePath)
marker = find_marker(image)
cm = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])

#draw a bounding box around the image and display it
box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
box = np.int0(box)
cv2.drawContours(image, [box], -1, (0,255,0), 2)
cv2.putText(image, "%.2fcm" % (cm), (image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
cv2.imshow("image", image)
cv2.waitKey(0)