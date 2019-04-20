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

from picamera.array import PiRGBArray
from picamera import PiCamera


from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS
import argparse
import imutils

SPEED = 0.3

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
                #processed_img = gray_image

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
                line_image = draw_lines(image, lines, thickness=1)
                #line_image = image
                return line_image
            

        
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
        
        


        last_time = time.time()
        '''
        while (True):
                rawCapture=PiRGBArray(camera, size=(800,600))
                #cap.framerate = 10
                camera.capture(rawCapture, format="bgr")
                frame=rawCapture.array
                #rawCapture = PiRGBArray(camera, size=(320, 240))
                #stream = camera.capture_continuous(rawCapture, format="bgr",
                #use_video_port=True)
                #frame = stream.array
                screen= process_img(frame)
                print('Frame took {} seconds'.format(time.time()-last_time))
                last_time = time.time()

                cv2.imshow('frame', screen)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        #cap.release()
        cv2.destroyAllWindows()
        
        '''
        '''
        # allow the camera to warmup and start the FPS counter
        print("[INFO] sampling frames from `picamera` module...")
        time.sleep(2.0)
        fps = FPS().start()
         
        # loop over some frames
        for (i, f) in enumerate(stream):
                # grab the frame from the stream and resize it to have a maximum
                # width of 400 pixels
                frame = f.array
                frame = imutils.resize(frame, width=400)
         
                # check to see if the frame should be displayed to our screen
                if args["display"] > 0:
                        cv2.imshow("Frame", frame)
                        key = cv2.waitKey(1) & 0xFF
         
                # clear the stream in preparation for the next frame and update
                # the FPS counter
                rawCapture.truncate(0)
                fps.update()
         
                # check to see if the desired number of frames have been reached
                if i == args["num_frames"]:
                        break
         
        # stop the timer and display FPS information
        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
         
        # do a bit of cleanup
        cv2.destroyAllWindows()
        stream.close()
        rawCapture.close()
        camera.close()
        '''
        
        # created a *threaded *video stream, allow the camera sensor to warmup,
        # and start the FPS counter
        print("[INFO] sampling THREADED frames from `picamera` module...")
        vs = PiVideoStream().start()
        time.sleep(2.0)
        #fps = FPS().start()
         
        # loop over some frames...this time using the threaded stream
        #while fps._numFrames < args["num_frames"]:
        '''
        while(True)
                # grab the frame from the threaded video stream and resize it
                # to have a maximum width of 800 pixels
                frame = vs.read()
                frame = imutils.resize(frame, width=800)
         
                # check to see if the frame should be displayed to our screen
                if args["display"] > 0:
                        cv2.imshow("Frame", frame)
                        key = cv2.waitKey(1) & 0xFF
         
                # update the FPS counter
                fps.update()
         
        # stop the timer and display FPS information
        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        '''
         
        # do a bit of cleanup

        while (True):
                #rawCapture=PiRGBArray(camera, size=(800,600))
                #cap.framerate = 10
                #camera.capture(rawCapture, format="bgr")
                frame=vs.read()
                frame = imutils.resize(frame, width=800)
                #rawCapture = PiRGBArray(camera, size=(320, 240))
                #stream = camera.capture_continuous(rawCapture, format="bgr",
                #use_video_port=True)
                #frame = stream.array
                screen= process_img(frame)
                print('Frame took {} seconds'.format(time.time()-last_time))
                last_time = time.time()

                cv2.imshow('frame', screen)
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


