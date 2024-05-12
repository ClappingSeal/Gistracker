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
from dronekit import connect
import logging
import keyboard
import os
import sys
import datetime
from collections import deque


class Variable:
    drone_lat = 35.2275168
    drone_lon = 126.8394166
    drone_height = 100

    pan_compensate = 0
    tilt_compensate = 0

    port_arduino = 'COM14'
    port_cube = 'COM5'
    ip_address = '192.168.1.3'


class Function:
    def your_MS(self, val, zoom):
        s = abs(val)
        if zoom <= 5:
            if s > 100:
                return 0.05 * val, 0.05 * s
            else:
                return 0.008 * val, 2
        elif 5 < zoom <= 15:
            if s > 150:
                return 0.03 * val, 0.03 * s
            else:
                return 0.005 * val, 1

        elif 15 < zoom <= 30:
            if s > 150:
                return 0.01 * val, 0.01 * s
            else:
                return 0.001 * val, 1

        elif 30 < zoom <= 50:
            if s > 200:
                return 0.002 * val, 0.01 * s
            else:
                return 0.001 * val, 1

        elif 50 < zoom <= 60:
            if s > 200:
                return 0.0012 * val, 0.006 * s
            else:
                return 0.0008 * val, 1

        elif 60 < zoom <= 61:
            if s > 250:
                return 0.0008 * val, 1
            else:
                return 0, 1

    def save_ptz_data_to_csv(self, yaw, pitch, zoom, w, h, name="ptz_data.csv"):
        file_exists = os.path.isfile(name)
        with open(name, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Timestamp", "Yaw", "Pitch", "Zoom", "Width", "Height"])
            writer.writerow([datetime.datetime.now(), yaw, pitch, zoom, w, h])

    def init_ptz_angle(self, camera_lat, camera_lon, drone_lat, drone_lon, drone_h, camera_heading):
        delta_lat = math.radians(drone_lat - camera_lat)
        delta_lon = math.radians(drone_lon - camera_lon)
        avg_lat = math.radians((camera_lat + drone_lat) / 2)
        delta_x = delta_lon * math.cos(avg_lat) * 6371000
        delta_y = delta_lat * 6371000

        d = math.sqrt(delta_x ** 2 + delta_y ** 2)
        pan_radians = math.atan2(delta_x, delta_y)
        pan_degrees = math.degrees(pan_radians)

        pan = (pan_degrees - camera_heading + 360) % 360
        if pan > 180:
            pan -= 360

        tilt = math.degrees(math.atan2(drone_h, d))

        if d >= 1000:
            zoom = 60
        elif 1000 > d >= 700:
            zoom = 50
        elif 700 > d >= 500:
            zoom = 40
        elif 500 > d >= 300:
            zoom = 30
        elif 300 > d >= 150:
            zoom = 20
        else:
            zoom = 10

        return pan, tilt, zoom


var = Variable()
fun = Function()
print(fun.init_ptz_angle(35.2266535, 126.8406746, 35.2271049, 126.8386817, var.drone_height, 270))
