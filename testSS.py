# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

camera = PiCamera();

camera.start_preview()
camera.capture('poza_test3.jpg')
camera.stop_preview()
