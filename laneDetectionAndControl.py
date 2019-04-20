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
import serial
from numpy import ones, vstack
from numpy.linalg import lstsq
from statistics import mean

from picamera.array import PiRGBArray
from picamera import PiCamera


from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS
import argparse
import imutils

SPEED = 0.1

#obstacle
waitline = b'OBST\r\n'
try:
        ser = serial.Serial("/dev/ttyUSB0", 9600)

except:
        print("Arduino nu e pe USB0")
if(ser==None):
        try:
                ser = serial.Serial("/dev/ttyUSB1", 9600)
        except:
                print("Arduino nu e pe USB1")




class PiVideoStream:
	def __init__(self, resolution=(800, 600), framerate=32):
		self.camera = PiCamera()
		self.camera.resolution = resolution
		self.camera.framerate = framerate
		self.rawCapture = PiRGBArray(self.camera, size=resolution)
		self.stream = self.camera.capture_continuous(self.rawCapture,
			format="bgr", use_video_port=True)
		self.frame = None
		self.stopped = False
		
	def start(self):
		Thread(target=self.update, args=()).start()
		return self
 
	def update(self):
		for f in self.stream:
			self.frame = f.array
			self.rawCapture.truncate(0)
			if self.stopped:
				self.stream.close()
				self.rawCapture.close()
				self.camera.close()
				return
	def read(self):
		return self.frame
 
	def stop(self):
		self.stopped = True



def cameraOptim():
        time.sleep(2)
        serialHandler = SerialHandler.SerialHandler("/dev/ttyACM0")
        serialHandler.startReadThread()
        serialHandler.sendPidActivation(True)

        def moveForward():
                serialHandler.sendMove(SPEED, -1.8)
                print("Forward")
        def moveLeft():
                serialHandler.sendMove(SPEED, -23.0)
                print("Left")
        def moveRight():
                serialHandler.sendMove(SPEED, 23.0)
                print("Right")
        def moveBackward():
                serialHandler.sendMove(-SPEED, -1.8)
                print("Back")
        def dontMove():
                serialHandler.sendMove(0,-1.8)
                print("Break")

        def region_of_interest(img, vertices):
                mask = np.zeros_like(img)
                match_mask_color = (255)
                cv2.fillPoly(mask, vertices, match_mask_color)
                masked_image = cv2.bitwise_and(img, mask)
                return masked_image


        def draw_lanes(img, lines, color=[0, 255, 255], thickness=3):
            # if this fails, go with some default line
            try:
                ys = []
                for i in lines:
                    for ii in i:
                        ys += [ii[1], ii[3]]
                min_y = min(ys)
                max_y = 600
                new_lines = []
                line_dict = {}

                for idx, i in enumerate(lines):
                    for xyxy in i:
                        x_coords = (xyxy[0], xyxy[2])
                        y_coords = (xyxy[1], xyxy[3])
                        A = vstack([x_coords, ones(len(x_coords))]).T
                        m, b = lstsq(A, y_coords)[0]
                        x1 = (min_y - b) / m
                        x2 = (max_y - b) / m

                        line_dict[idx] = [m, b, [int(x1), min_y, int(x2), max_y]]
                        new_lines.append([int(x1), min_y, int(x2), max_y])

                final_lanes = {}

                for idx in line_dict:
                    final_lanes_copy = final_lanes.copy()
                    m = line_dict[idx][0]
                    b = line_dict[idx][1]
                    line = line_dict[idx][2]

                    if len(final_lanes) == 0:
                        final_lanes[m] = [[m, b, line]]

                    else:
                        found_copy = False

                        for other_ms in final_lanes_copy:

                            if not found_copy:
                                if abs(other_ms * 1.2) > abs(m) > abs(other_ms * 0.8):
                                    if abs(final_lanes_copy[other_ms][0][1] * 1.2) > abs(b) > abs(
                                            final_lanes_copy[other_ms][0][1] * 0.8):
                                        final_lanes[other_ms].append([m, b, line])
                                        found_copy = True
                                        break
                                else:
                                    final_lanes[m] = [[m, b, line]]

                line_counter = {}

                for lanes in final_lanes:
                    line_counter[lanes] = len(final_lanes[lanes])

                top_lanes = sorted(line_counter.items(), key=lambda item: item[1])[::-1][:2]

                lane1_id = top_lanes[0][0]
                lane2_id = top_lanes[1][0]

                def average_lane(lane_data):
                    x1s = []
                    y1s = []
                    x2s = []
                    y2s = []
                    for data in lane_data:
                        x1s.append(data[2][0])
                        y1s.append(data[2][1])
                        x2s.append(data[2][2])
                        y2s.append(data[2][3])
                    return int(mean(x1s)), int(mean(y1s)), int(mean(x2s)), int(mean(y2s))

                l1_x1, l1_y1, l1_x2, l1_y2 = average_lane(final_lanes[lane1_id])
                l2_x1, l2_y1, l2_x2, l2_y2 = average_lane(final_lanes[lane2_id])

                return [l1_x1, l1_y1, l1_x2, l1_y2], [l2_x1, l2_y1, l2_x2, l2_y2], lane1_id, lane2_id
            except Exception as e:
                print(str(e))
            
        def process_img(image):
                original_image = image
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                processed_img = cv2.Canny(gray_image, 200, 300)
                processed_img = cv2.GaussianBlur(processed_img, (5, 5), 0)

                vertices = np.array([[10, 500], [10, 400], [300, 250], [500, 250], [800, 500], [800, 500],
                                 ], np.int32)
                cropped_image = region_of_interest(processed_img, [vertices])
                lines = cv2.HoughLinesP(
                        cropped_image,
                        rho=1,
                        theta=np.pi / 180,
                        threshold=180,
                        lines=np.array([]),
                        minLineLength=20,
                        maxLineGap=15
                )
                m1 = 0
                m2 = 0
                try:
                        l1, l2, m1, m2 = draw_lanes(original_image, lines)
                        cv2.line(original_image, (l1[0], l1[1]), (l1[2], l1[3]), [0, 255, 0], 30)
                        cv2.line(original_image, (l2[0], l2[1]), (l2[2], l2[3]), [0, 255, 0], 30)
                except Exception as e:
                        print(str(e))
                        pass
                try:
                        for coords in lines:
                            coords = coords[0]
                            try:
                                cv2.line(processed_img, (coords[0], coords[1]), (coords[2], coords[3]), [255, 0, 0], 3)

                            except Exception as e:
                                print(str(e))
                except Exception as e:
                        pass
                return processed_img, original_image, m1, m2
            
        ap = argparse.ArgumentParser()
        ap.add_argument("-n", "--num-frames", type=int, default=100,
                help="# of frames to loop over for FPS test")
        ap.add_argument("-d", "--display", type=int, default=1,
                help="Whether or not frames should be displayed")
        args = vars(ap.parse_args())
         

        print("[INFO] sampling THREADED frames from `picamera` module...")
        vs = PiVideoStream().start()

        last_time = time.time()
        while (True):
                frame=vs.read()
                frame = imutils.resize(frame, width=800)
                last_time = time.time()
                new_screen,original_image, m1, m2 = process_img(frame)
                print('Frame took {} seconds'.format(time.time()-last_time))


                cv2.imshow('frame', original_image) #se mai poate adauga argumentul cv2.COLOR_BGR2RGB

                if(waitline != ser.readline()):
                
                        if m1 < 0 and m2 < 0:
                                moveRight()
                                time.sleep(0.1)
                                #dontMove()
                        elif m1 > 0 and m2 > 0:
                                moveLeft()
                                time.sleep(0.1)
                                #dontMove()
                        else:
                                moveForward()
                                time.sleep(0.1)
                                #dontMove()
                else:
                        dontMove()
                        print("OBSTACLE AHEAD")
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cv2.destroyAllWindows()
        vs.stop()
        


    
try:
        t4 = threading.Thread(name='CameraOptim', target = cameraOptim)
        t4.start() 
except:
   print("Error: unable to start thread")


