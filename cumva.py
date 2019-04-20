from __future__ import print_function
import sys
sys.path.append('../')
from rpi import SerialHandler
import threading
from threading import Thread
import time
import cv2
import pyglet
import numpy as np
from numpy import ones, vstack
from numpy.linalg import lstsq
from statistics import mean

from picamera.array import PiRGBArray
from picamera import PiCamera


from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS
import argparse
import imutils

class PiVideoStream:
	def __init__(self, resolution=(800, 600), framerate=32):
		# initialize the camera and stream
		self.camera = PiCamera()
		self.camera.resolution = resolution
		self.camera.framerate = framerate
		self.rawCapture = PiRGBArray(self.camera, size=resolution)
		self.stream = self.camera.capture_continuous(self.rawCapture,
			format="bgr", use_video_port=True)
 
		# initialize the frame and the variable used to indicate
		# if the thread should be stopped
		self.frame = None
		self.stopped = False
		
	def start(self):
		# start the thread to read frames from the video stream
		Thread(target=self.update, args=()).start()
		return self
 
	def update(self):
		# keep looping infinitely until the thread is stopped
		for f in self.stream:
			# grab the frame from the stream and clear the stream in
			# preparation for the next frame
			self.frame = f.array
			self.rawCapture.truncate(0)
 
			# if the thread indicator variable is set, stop the thread
			# and resource camera resources
			if self.stopped:
				self.stream.close()
				self.rawCapture.close()
				self.camera.close()
				return
	def read(self):
                # return the frame most recently read
		return self.frame
 
	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True
def lineDet(image):
    # grab the image dimensions
    h = image.shape[0]
    w = image.shape[1]
    a= [0]
    b= [0]
    i= 0
    x, y = [int(w/2),int(w/2)]
    ok1, ok2= [0, 0]
    #for z in range(h-1, 0, -1):
    while (ok1 < 1) or (ok2 < 1):
            if ok1 < 1:
                if image[h-1, x] >= 200:
                    a[i] = x
                    ok1 = ok1 + 1
            if ok2 < 1:
                if image[h-1, y] >= 200:
                    b[i] = y
                    ok2 = ok2 + 1
            #if a[i-1]-5 <= x:
            x = x - 1
            #else:
                #ok1 = 1
                #a[i]=a[i-1]+1
            #if b[i-1]+5 <= y:
            y = y + 1
            #else:
                #ok2 = 1
                #b[i]=i-1
        #i=i+1
        #x, y =[int(w / 2), int(w / 2)]
        #ok1, ok2 = [0, 0]
    return a,b

vs = PiVideoStream().start()
time.sleep(2.0)

last_time = time.time()
while True:
    
    frame=vs.read()
    frame = imutils.resize(frame, width=800)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    last_time = time.time()
    h1=img.shape[0]
    w1=img.shape[1]
    roi= img[400:h1, 0:w1]
    mat,mat1=lineDet(roi)
    n=len(mat)-1
    m=len(mat1)-1
    cv2.line(img, (mat[0],img.shape[0]), (mat[n], img.shape[0]-n), (0, 0, 255), 15)
    cv2.line(img, (mat1[0],img.shape[0]), (mat1[m], img.shape[0]-m), (0, 0, 255), 15)
    print('Frame took {} seconds'.format(time.time()-last_time))
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
vs.stop()
