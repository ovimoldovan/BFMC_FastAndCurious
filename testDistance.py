import serial
import RPi.GPIO as GPIO
import time

ser = serial.Serial("/dev/ttyUSB0", 9600)

ser.baudrate=9600
print("a")

waitline = b'OBST\r\n'

while 1:
    line = ser.readline()
    if(line == waitline):
        print(1)

