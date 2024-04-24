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


class PTZ:
    def __init__(self):
        print("connecting ptz...")
        self.port = 'COM5'
        self.baud_rate = 115200
        self.ser = serial.Serial(self.port, self.baud_rate)

        self.ip_address = '192.168.1.3'
        self.username = 'admin'
        self.password = '123456'
        self.ptz_control_url = f'http://{self.ip_address}/cgi-bin/getuid?username={self.username}&password={self.password}'

        self.response = requests.get(
            self.ptz_control_url,
            auth=HTTPDigestAuth(self.username, self.password)
        )
        self.root = ET.fromstring(self.response.text)
        self.uid_text = self.root.text

        self.yaw = 0
        self.pitch = 0

    def set_motor(self, angle1, angle2, velocity1, velocity2):
        command = f"{angle1} {angle2} {velocity1} {velocity2}\n"
        self.ser.write(command.encode())

    def zoom(self, num):
        ptz_control_url = f'http://{self.ip_address}/cgi-bin/ptz_ctrl?call_preset={num}&uid={self.uid_text}'
        response = requests.get(
            ptz_control_url,
            auth=HTTPDigestAuth(self.username, self.password)
        )

    def focus(self, num):
        ptz_control_url = f'http://{self.ip_address}/cgi-bin/ptz_ctrl?focus={num}&uid={self.uid_text}'
        response = requests.get(
            ptz_control_url,
            auth=HTTPDigestAuth(self.username, self.password)
        )

    def yaw_pitch(self, yaw, pitch, yaw_speed, pitch_speed):
        if yaw > 90:
            yaw = 90
        if yaw < -90:
            yaw = -90
        if pitch > 60:
            pitch = 60
        if pitch < 0:
            pitch = 0

        yaw = 148 - yaw
        pitch = 148 + pitch
        self.set_motor(yaw, pitch, yaw_speed, pitch_speed)

    def get_angle(self):
        self.ser.write(b'1\n')
        time.sleep(0.1)

        if self.ser.inWaiting() > 0:
            data = self.ser.readline().decode().strip()
            angles = data.split(' ')
            if len(angles) == 2:
                self.yaw = 146.5 - int(angles[0])
                self.pitch = -147 + int(angles[1])

        return self.yaw, self.pitch


class VISION:
    def __init__(self):
        print("connecting camera...")
        self.rtsp_url = 'rtsp://admin:123456@192.168.1.3/stream0'
        self.frame = None
        self.stream_thread = threading.Thread(target=self.rtsp_stream_handler)

        self.stream_thread.start()

    def rtsp_stream_handler(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                cap = cv2.VideoCapture(self.rtsp_url)
                continue

            self.frame = cv2.resize(frame, (1920, 960))

        cap.release()


# not use
class Detect1:
    def __init__(self, vision_instance):
        self.vision = vision_instance
        self.dark_threshold = 200
        self.ignore_region = (1800, 850, 250, 120)

    def is_within_ignore_region(self, x, y, w, h):
        center_x, center_y, ignore_w, ignore_h = self.ignore_region
        ignore_x1 = center_x - ignore_w // 2
        ignore_y1 = center_y - ignore_h // 2
        ignore_x2 = center_x + ignore_w // 2
        ignore_y2 = center_y + ignore_h // 2

        if x < ignore_x2 and x + w > ignore_x1 and y < ignore_y2 and y + h > ignore_y1:
            return True
        return False

    def process_frame(self, frame):
        frame = cv2.resize(frame, dsize=(1920, 960), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opened = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        largest_area = 0
        largest_rect = None
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            rect_area = w * h
            if 4 < rect_area < 400:
                area = cv2.contourArea(contour)
                if 4 < area < 10000 and not self.is_within_ignore_region(x, y, w, h):
                    if area > largest_area:
                        mask = np.zeros(gray.shape, dtype=np.uint8)
                        mean_val = cv2.mean(frame, mask=mask)[0]

                        if mean_val <= self.dark_threshold:
                            largest_area = area
                            largest_rect = (x, y, w, h)

        if largest_rect is not None:
            x, y, w, h = largest_rect
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return frame, x, y, w, h

        return frame, 960, 480, 0, 0


class Detect2:
    def __init__(self, vision_instance):
        self.vision = vision_instance
        self.ignore_region = (1800, 850, 250, 120)

    def is_within_ignore_region(self, x, y, w, h):
        center_x, center_y, ignore_w, ignore_h = self.ignore_region
        ignore_x1 = center_x - ignore_w // 2
        ignore_y1 = center_y - ignore_h // 2
        ignore_x2 = center_x + ignore_w // 2
        ignore_y2 = center_y + ignore_h // 2

        return x < ignore_x2 and x + w > ignore_x1 and y < ignore_y2 and y + h > ignore_y1

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = []
        sizes = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:
                x, y, w, h = cv2.boundingRect(contour)
                if not self.is_within_ignore_region(x, y, w, h):
                    centers.append((x + w // 2, y + h // 2))
                    sizes.append((w, h))

        if centers:
            avg_x = sum([c[0] for c in centers]) // len(centers)
            avg_y = sum([c[1] for c in centers]) // len(centers)
            min_size = max(sizes, key=lambda size: size[0] * size[1])
            width, height = min_size
            # cv2.rectangle(frame, (avg_x - width // 2, avg_y - height // 2), (avg_x + width // 2, avg_y + height // 2),
            #               (0, 255, 0), 2)
            return frame, x, y, w, h

        return frame, 960, 480, 0, 0


class Detect3:
    def __init__(self, vision_instance):
        self.vision = vision_instance
        # Setup SimpleBlobDetector parameters.
        self.params = cv2.SimpleBlobDetector_Params()
        self.params.minThreshold = 10
        self.params.maxThreshold = 200
        self.params.filterByArea = True
        self.params.minArea = 10
        self.params.maxArea = 1000
        self.detector = cv2.SimpleBlobDetector_create(self.params)
        self.ignore_region = (1800, 850, 180, 90)

    def is_within_ignore_region(self, x, y, w, h):
        center_x, center_y, ignore_w, ignore_h = self.ignore_region
        ignore_x1 = center_x - ignore_w // 2
        ignore_y1 = center_y - ignore_h // 2
        ignore_x2 = center_x + ignore_w // 2
        ignore_y2 = center_y + ignore_h // 2

        return x < ignore_x2 and x + w > ignore_x1 and y < ignore_y2 and y + h > ignore_y1

    def process_frame(self, frame):
        # Detect blobs
        keypoints = self.detector.detect(frame)
        if keypoints:
            largest_keypoint = max(keypoints, key=lambda keypoint: keypoint.size)
            keypoints = [largest_keypoint]  # Only keep the largest keypoint

        # Draw detected blobs as red circles
        im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            w, h = int(kp.size), int(kp.size)
            if not self.is_within_ignore_region(x, y, w,ㅘ
