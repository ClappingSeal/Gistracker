import serial
import time
import requests
from requests.auth import HTTPDigestAuth
import xml.etree.ElementTree as ET
import cv2
import threading
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from operator import add
import tensorflow as tf
import joblib
import warnings
import csv
import os
import math
from dronekit import connect, VehicleMode, Command, LocationGlobalRelative
from pymavlink import mavutil
import logging

logging.getLogger('dronekit').setLevel(logging.CRITICAL)


class Variable:
    port_arduino = 'COM14'
    port_cube = 'COM5'
    ip_address = '192.168.1.3'


class CubeOrange:
    def __init__(self):
        self.connection_string = Variable.port_cube
        self.baudrate = 115200
        self.vehicle = connect(self.connection_string, wait_ready=True, baud=self.baudrate, timeout=100)

    def get_pos(self):
        init_lat = self.vehicle.location.global_relative_frame.lat
        init_lon = self.vehicle.location.global_relative_frame.lon

        if init_lat == 0 or init_lon == 0:
            print("Go outside!")

        return init_lat, init_lon

    def get_direction(self):
        heading = self.vehicle.heading
        return heading


class PTZ:
    def __init__(self):
        print("connecting ptz...")
        self.port = Variable.port_arduino
        self.baud_rate = 115200
        self.ser = serial.Serial(self.port, self.baud_rate)

        self.yaw = 0
        self.pitch = 0

    def set_motor(self, angle1, angle2, velocity1, velocity2):
        command = f"{angle1} {velocity1} {angle2} {velocity2}\n"
        self.ser.write(command.encode())

    def yaw_pitch(self, yaw, pitch, yaw_speed, pitch_speed):
        if yaw > 90:
            yaw = 90
        if yaw < -90:
            yaw = -90

        yaw = int(((180 - yaw) / 90 * 1024))
        pitch = int(2048 + pitch / 90 * 1024)

        self.set_motor(yaw, pitch, yaw_speed, pitch_speed)

    def get_angle(self):
        self.ser.write(b'1\n')
        time.sleep(0.1)

        if self.ser.inWaiting() > 0:
            data = self.ser.readline().decode().strip()
            angles = data.split(' ')

            if len(angles) == 2:
                self.yaw = (2048 - int(angles[0])) / 1024 * 90
                self.pitch = (-2048 + int(angles[1])) / 1024 * 90

        return self.yaw, self.pitch


ptz = PTZ()
for i in range (0,10):
    ptz.yaw_pitch(0, 0, 50, 50)
    time.sleep(2)
    ptz.yaw_pitch(0, 90, 50, 50)
    time.sleep(2)

print(ptz.get_angle())




#############

import cv2
from ultralytics import YOLO
from operator import add

params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10
params.maxThreshold = 200

# Filter by Area.
params.filterByArea = True
params.minArea = 10
params.maxArea = 1000
# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.01

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

model = YOLO('2class.pt')
classes = {0: 'fixed', 1: 'quad'}
# Open video file or camera
video = cv2.VideoCapture('aa.avi')  # Replace 'video.mp4' with the path to your video file
class_name = "Detecting"
avg = [0, 0, 0]
while True:
    # Read a frame from the video
    ret, frame = video.read()

    # Check if the frame was successfully read
    if not ret:
        break

    largest_rect = None

    keypoints = detector.detect(frame)
    if keypoints:
        largest_keypoint = max(keypoints, key=lambda keypoint: keypoint.size)
        # print(largest_keypoint.size)
        # print(keypoints)
        # print(largest_keypoint)
        # keypoints = [largest_keypoint]  # Only keep the largest keypoint
        # xy = cv2.KeyPoint_convert(largest_keypoint)[0]
        xy = largest_keypoint.pt
        # print(int(largest_keypoint.size))
        r = int(largest_keypoint.size)
        largest_rect = (int(xy[0]) - r - 5, int(xy[1]) - r - 5, r * 2 + 10, r * 2 + 10)

    if largest_rect is not None:
        x, y, w, h = largest_rect
        # Crop the rectangle area
        cropped = frame[(y - 20):(y + h + 20), (x - 20):(x + w + 20)]
        if cropped.size == 0:
            continue
        detection = model(cropped, verbose=False)[0]
        probs = list(detection.probs.data.tolist())
        if max(probs) > 0.6:
            probs.append(1)
            avg = list(map(add, avg, probs))
        # print(avg)
        if avg[2] == 10:
            highest_prob = max(avg[:2])
            highest_prob_index = avg.index(highest_prob)
            class_name = classes[highest_prob_index]
            avg = [0, 0, 0]
        cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        # Perform further processing on the cropped area

        # Draw bounding rectangle on the original frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Video', frame)

    # Check for key press to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture object and close windows
video.release()
cv2.destroyAllWindows()