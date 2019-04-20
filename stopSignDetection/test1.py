import numpy as np
import cv2

stop_cascade = cv2.CascadeClassifier('data/cascade.xml')

#img = cv2.imread('stop6.jpg')

#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#stop = stop_cascade.detectMultiScale(gray, 1.1, 10)

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    stop = stop_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in stop:
        # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Stop', (x + w, y + h), font, 0.75, (11, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('img',img)

    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()