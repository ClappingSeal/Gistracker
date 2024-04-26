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
    port_arduino = 'COM3'
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


cubeorange = CubeOrange()
while True:
    print(cubeorange.get_direction())
