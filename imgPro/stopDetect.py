import cv2
import numpy as np

img_rgb = cv2.imread('img/stopSign.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

#template = cv2.imread('opencv-template-for-matching.jpg',0)
camera = cv2.VideoCapture(0)
template = cv2.imread(camera,0)
w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)




#cv2.imshow('Detected',img_rgb)

if camera.isOpened():
    while (True):
        # cv2.waitKey(0)

        value, image = camera.read()



        #cv2.imwrite("img/distanceTest/aopencv" + str(0) + '.jpg', image)
        cv2.imshow("image", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            camera.release()
            break
else:
    print("No camera")