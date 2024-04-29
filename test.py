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
    port_arduino = 'COM10'
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
ptz.yaw_pitch(80, 0.2, 50, 50)
time.sleep(2)
print(ptz.get_angle())
