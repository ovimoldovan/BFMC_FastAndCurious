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

SPEED = 0.2

stop_cascade = cv2.CascadeClassifier('stopSignDetection/data/cascade.xml')

''' INCEPUT CAMERA OPTIM '''

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

                # Define a blank matrix that matches the image height/width.
                mask = np.zeros_like(img)
                # Retrieve the number of color channels of the image.
                #channel_count = img.shape[2]
                # Create a match color with the same color channel counts.
                match_mask_color = (255)
                # Fill inside the polygon
                cv2.fillPoly(mask, vertices, match_mask_color)
                # Returning the image only where mask pixels match
                masked_image = cv2.bitwise_and(img, mask)
                return masked_image


        def draw_lanes(img, lines, color=[0, 255, 255], thickness=3):
            # if this fails, go with some default line
            try:

                # finds the maximum y value for a lane marker
                # (since we cannot assume the horizon will always be at the same point.)

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
                        # These four lines:
                        # modified from http://stackoverflow.com/questions/21565994/method-to-return-the-equation-of-a-straight-line-given-two-points
                        # Used to calculate the definition of a line, given two sets of coords.
                        x_coords = (xyxy[0], xyxy[2])
                        y_coords = (xyxy[1], xyxy[3])
                        A = vstack([x_coords, ones(len(x_coords))]).T
                        m, b = lstsq(A, y_coords)[0]

                        # Calculating our new, and improved, xs
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
                #processed_img = gray_image

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
                #line_image = draw_lines(image, lines, thickness=1)
                #line_image = image
                #return line_image
                return processed_img, original_image, m1, m2
            

        
        # construct the argument parse and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-n", "--num-frames", type=int, default=100,
                help="# of frames to loop over for FPS test")
        ap.add_argument("-d", "--display", type=int, default=1,
                help="Whether or not frames should be displayed")
        args = vars(ap.parse_args())
         
        # initialize the camera and stream
        #camera = PiCamera()
        #camera.resolution = (800, 600)
        #camera.framerate = 32
        
        


        #last_time = time.time()

        
        # created a *threaded *video stream, allow the camera sensor to warmup,
        # and start the FPS counter
        print("[INFO] sampling THREADED frames from `picamera` module...")
        vs = PiVideoStream().start()
        time.sleep(2.0)
        #fps = FPS().start()
         
        # loop over some frames...this time using the threaded stream
        #while fps._numFrames < args["num_frames"]:

        # do a bit of cleanup
        last_time = time.time()
        while (True):
                #rawCapture=PiRGBArray(camera, size=(800,600))
                #cap.framerate = 10
                #camera.capture(rawCapture, format="bgr")
                frame=vs.read()
                frame = imutils.resize(frame, width=800)
                last_time = time.time()
                new_screen,original_image, m1, m2 = process_img(frame)
                #rawCapture = PiRGBArray(camera, size=(320, 240))
                #stream = camera.capture_continuous(rawCapture, format="bgr",
                #use_video_port=True)
                #frame = stream.array
                #screen= process_img(frame)
                print('Frame took {} seconds'.format(time.time()-last_time))

                ''' STOP SIGN '''

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                vertices = np.array([[600, 10], [600, 300], [800, 300], [800, 500], [800, 10],
                                 ], np.int32)
                stop_semn = region_of_interest(gray, [vertices])
                
                stop = stop_cascade.detectMultiScale(stop_semn, 1.3, 5)
                

                
                for (x, y, w, h) in stop:
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame, 'Stop', (x + w, y + h), font, 0.75, (11, 11, 255), 2, cv2.LINE_AA)
                        print("STOP")
                        time.sleep(10)

                cv2.imshow('frame', original_image) #se mai poate adauga argumentul cv2.COLOR_BGR2RGB


                if m1 < 0 and m2 < 0:
                        moveRight()
                        time.sleep(0.1)
                        dontMove()
                elif m1 > 0 and m2 > 0:
                        moveLeft()
                        time.sleep(0.1)
                        dontMove()
                else:
                        moveForward()
                        time.sleep(0.1)
                        dontMove()

                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cv2.destroyAllWindows()
        vs.stop()
        

''' FINAL CAMERA OPTIM '''


def keyboard():
    
    time.sleep(2)
    serialHandler = SerialHandler.SerialHandler("/dev/ttyACM0")
    serialHandler.startReadThread()
    serialHandler.sendPidActivation(True)
    
    window = pyglet.window.Window(width=360, height=200)

    @window.event
    def on_key_press(key,modifiers):
        direction= -1
        if (key == pyglet.window.key.UP):
            direction = 1
            print("MOVE UP +{} ",direction)
            serialHandler.sendMove(SPEED, -1.8)   
        elif (key == pyglet.window.key.DOWN):
            direction = -1
            print("MOVE DOWN +{} ", direction)
            serialHandler.sendMove(-SPEED, -1.8)
            #time.sleep(0.2)
            #serialHandler.sendBrake(0.1)
        elif (key == pyglet.window.key.LEFT):
            #serialHandler.sendBezierCurve(1.0+1.0j, 1.56+0.44j, 1.56-0.44j, 1.0-1.0j, 3.0, True)
            serialHandler.sendMove( SPEED * direction,  -23.0)
            print("MOVE LEFT")
        elif (key == pyglet.window.key.RIGHT):
            #serialHandler.sendBezierCurve(1.0-1.0j, 1.56-0.44j, 1.56+0.44j, 1.0+1.0j, 3.0, True)
            #serialHandler.sendBezierCurve(0.0+0.0j, 0.0+0.333j, 0.0+0.66j, 0.0+1.0j, 0.1, True)
            serialHandler.sendMove(SPEED * direction, 23.0)
            print("MOVE RIGHT")
        elif(key == pyglet.window.key.SPACE):
            #time.sleep(0.5)
            serialHandler.sendBrake(0.1)
    pyglet.app.run()

def testRun():
    time.sleep(2)
    serialHandler = SerialHandler.SerialHandler("/dev/ttyACM0")
    serialHandler.startReadThread()
    serialHandler.sendPidActivation(True)
    serialHandler.sendMove(0.1, 8.5)
    time.sleep(12)
    serialHandler.sendBrake(8.5)
    #serialHandler.sendBezierCurve(1.0+1.0j, 1.56+0.44j, 1.56-0.44j, 1.0-1.0j, 3.0, True)

def startRecording():
    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 10
    rawCapture = PiRGBArray(camera, size=(640, 480))
    #out = cv2.VideoWriter('/home/pi/Desktop/output',cv2.cv.CV_FOURCC('M','J','P','G'), 20.0, (640,480))
    out = cv2.VideoWriter('/home/pi/Desktop/testvideo.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 10.0, 
               (640,480),True)
    # allow the camera to warmup
    time.sleep(0.1)

    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        image = frame.array

        # show the frame
        cv2.imshow("Frame", image)
        key = cv2.waitKey(1) & 0xFF
        out.write(image)
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            out.release()
            #camera.release()
            cv2.destroyAllWindows()
            break
    #out.release()

def realTimeLineDetection():
        cap = PiCamera()
    
        def region_of_interest(img, vertices):

                # Define a blank matrix that matches the image height/width.
                mask = np.zeros_like(img)
                # Retrieve the number of color channels of the image.
                #channel_count = img.shape[2]
                # Create a match color with the same color channel counts.
                match_mask_color = (255)
                # Fill inside the polygon
                cv2.fillPoly(mask, vertices, match_mask_color)
                # Returning the image only where mask pixels match
                masked_image = cv2.bitwise_and(img, mask)
                return masked_image


        def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
            if lines is None:
                    return img
            line_img = np.zeros(
                (
                    img.shape[0],
                    img.shape[1],
                    3
                ),
                dtype=np.uint8,
            )
            # Loop over all lines and draw them on the blank image.
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
            # Merge the image with the lines onto the original.
            img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
            # Return the modified image.
            return img


    
        def process_img(image):
                gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                processed_img = cv2.Canny(gray_image, 200, 300)

                vertices = np.array([[10, 500], [10, 400], [300, 250], [500, 250], [800, 500], [800, 500],
                                 ], np.int32)
                cropped_image = region_of_interest(
                        processed_img,
                        np.array([vertices], np.int32)
                        )
                lines = cv2.HoughLinesP(
                        cropped_image,
                        rho=2,
                        theta=np.pi / 180,
                        threshold=40,
                        lines=np.array([]),
                        minLineLength=10,
                        maxLineGap=110
                )
                line_image = draw_lines(image, lines, thickness=10)
                #line_image = image
                return line_image
    
        last_time = time.time()

        cap.resolution = (640, 480)
        cap.framerate = 30
        
        while (True):
                rawCapture=PiRGBArray(cap)
                cap.capture(rawCapture, format="bgr")
                frame=rawCapture.array
                screen= process_img(frame)
                print('Frame took {} seconds'.format(time.time()-last_time))
                last_time = time.time()

                cv2.imshow('frame', screen)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        #cap.release()
        cv2.destroyAllWindows()

    
try:
    #t1 = threading.Thread(name='recorder', target = startRecording)
    #t2 = threading.Thread(name='runner', target = testRun)
        #t2= threading.Thread(name='keyboardRunner', target = keyboard)
    #t3= threading.Thread(name='liveLaneDetection', target=realTimeLineDetection)
    #t1.start()
    #t1.sleep(2)
        #t2.start()
    #
    #t3.start()
        t4 = threading.Thread(name='CameraOptim', target = cameraOptim)
        t4.start() 
except:
   print("Error: unable to start thread")


