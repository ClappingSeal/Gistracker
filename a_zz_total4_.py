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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('dronekit').setLevel(logging.CRITICAL)


# 추후 ptz 보정 필요
# 추후 초점 프로토콜 1~4 선택
# 날씨에 따라 잘 되는 detect 버전에 따라 expand, reduce area 설정
# 혹시 모르니 ptz_data.csv 삭제 후 시작

class Variable:
    drone_lat = 35.2268748
    drone_lon = 126.840071
    drone_height = 98

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


class PTZ:
    def __init__(self):
        print("connecting ptz...")
        self.port = Variable.port_arduino
        self.baud_rate = 115200
        self.ser = serial.Serial(self.port, self.baud_rate)

        self.ip_address = Variable.ip_address
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
        command = f"{angle1} {velocity1} {angle2} {velocity2}\n"
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
        yaw = yaw % 360
        if yaw > 180:
            yaw -= 360

        if yaw > 90:
            yaw = 90
        if yaw < -90:
            yaw = -90

        if pitch > 60:
            pitch = 60
        if pitch < 0:
            pitch = 0

        yaw = int(((180 - yaw) / 90 * 1024))
        pitch = int(2048 + pitch / 90 * 1024)
        yaw_speed = int(yaw_speed)
        pitch_speed = int(pitch_speed)

        self.set_motor(yaw, pitch, yaw_speed, pitch_speed)

    def get_angle(self):
        self.ser.write(b'1\n')
        time.sleep(0.01)

        if self.ser.inWaiting() > 0:
            data = self.ser.readline().decode().strip()
            angles = data.split(' ')

            if len(angles) == 2:
                self.yaw = (2048 - int(angles[0])) / 1024 * 90
                self.pitch = (-2048 + int(angles[1])) / 1024 * 90

        return self.yaw, self.pitch


class VISION:
    def __init__(self):
        print("connecting camera...")
        self.rtsp_url = f'rtsp://admin:123456@{Variable.ip_address}/stream0'
        self.frame = None
        self.stream_thread = threading.Thread(target=self.rtsp_stream_handler)

        warnings.filterwarnings("ignore", category=UserWarning, module="libav")
        logging.getLogger('libav').setLevel(logging.ERROR)

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


class CubeOrange:
    def __init__(self):
        print("connecting cube")
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
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
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
            cv2.rectangle(frame, (avg_x - width // 2, avg_y - height // 2), (avg_x + width // 2, avg_y + height // 2),
                          (0, 255, 0), 2)
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
        self.params.minArea = 20
        self.params.maxArea = 1000

        self.params.filterByCircularity = False
        self.params.minCircularity = 0.1
        self.params.filterByConvexity = False
        self.params.minConvexity = 0.87
        self.params.filterByInertia = False
        self.params.minInertiaRatio = 0.01

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

            xy = largest_keypoint.pt
            r = int(largest_keypoint.size)
            largest_rect = (int(xy[0]) - r - 5, int(xy[1]) - r - 5, r * 2 + 10, r * 2 + 10)
            x, y, w, h = largest_rect

            if not self.is_within_ignore_region(x, y, w, h):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                return frame, x, y, w, h
            else:
                return frame, 960, 480, 0, 0
        else:
            return frame, 960, 480, 0, 0


class DetectPink:
    def __init__(self, vision_instance):
        self.vision = vision_instance

    def process_frame(self, frame):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_pink = np.array([145, 100, 100])
        upper_pink = np.array([175, 255, 255])
        pink_mask = cv2.inRange(hsv_frame, lower_pink, upper_pink)

        kernel = np.ones((5, 5), np.uint8)
        pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return frame, x, y, w, h

        return frame, 960, 480, 0, 0


class Recognize:
    def __init__(self):
        self.model = YOLO('2class.pt')
        self.classes = {0: 'fixed', 1: 'quad'}
        self.class_name = "Detecting..."
        self.avg = [0, 0, 0]

    def bounding_box(self, frame, x, y, w, h):
        cropped = frame[(y - 30):(y + h + 30), (x - 30):(x + w + 30)]

        if cropped.shape[0] > 0 and cropped.shape[1] > 0:
            detection = self.model(cropped, verbose=False)[0]
            probs = list(detection.probs.data.tolist())
            probs.append(1)
            self.avg = list(map(add, self.avg, probs))
            if self.avg[2] == 10:
                highest_prob = max(self.avg[:2])
                highest_prob_index = self.avg.index(highest_prob)
                self.class_name = self.classes[highest_prob_index]
                self.avg = [0, 0, 0]
            return self.class_name


class LSTM:
    def __init__(self, scaler_path, model_path):
        warnings.filterwarnings("ignore", category=UserWarning)  # 경고 무시
        self.scaler = joblib.load(scaler_path)
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, test_data):
        test_data = np.array(test_data, dtype=np.float32).reshape(-1, 2)
        scaled_test_data = self.scaler.transform(test_data)
        scaled_test_data = scaled_test_data.reshape(1, -1, 2)
        self.interpreter.set_tensor(self.input_details[0]['index'], scaled_test_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        predicted_position = self.scaler.inverse_transform(output_data)

        return predicted_position


class Memorize:
    def __init__(self, size=5):
        self.size = size
        self.queue = []

    def add(self, x, y):
        if len(self.queue) >= self.size:
            self.queue.pop(0)
        self.queue.append([x, y])

    def get_all(self):
        return self.queue


class SetZoom:
    def __init__(self):
        self.zoom_array1 = [1, 3, 5, 8, 10, 13, 15, 20, 25, 30, 40, 50, 60, 61]
        self.zoom_array2 = [1, 3, 5, 8, 10, 13, 15, 20, 25, 30, 40, 50, 60, 120]

        # self.zoom_array1 = [15, 30, 60, 61, 62]
        # self.zoom_array2 = [15, 30, 60, 120, 240]

        self.expanding_area = 200
        self.reducing_area = 2000

    def change_zoom(self, w, h, current_zoom):
        area = w * h
        # reducing
        if area > self.reducing_area or area == 0:
            indices = np.where(np.array(self.zoom_array1) == current_zoom)[0]
            if current_zoom == 5:
                return self.zoom_array1[int(indices[0])]
            indices = np.where(np.array(self.zoom_array1) == current_zoom)[0]
            return self.zoom_array1[int(indices[0]) - 1]

        # expanding
        if area < self.expanding_area:
            indices = np.where(np.array(self.zoom_array1) == current_zoom)[0]
            if current_zoom == 61:
                return self.zoom_array1[int(indices[0])]
            else:
                return self.zoom_array1[int(indices[0]) + 1]

        else:
            indices = np.where(np.array(self.zoom_array1) == current_zoom)[0]
            return self.zoom_array1[int(indices[0])]


if __name__ == "__main__":
    # region 1. class declare
    ptz = PTZ()
    vision = VISION()
    cube = CubeOrange()
    detect = Detect3(vision)
    recognize = Recognize()
    lstm = LSTM('scaler.pkl', 'lstm_drone_positions_model.tflite')
    memorize = Memorize()
    set_zoom = SetZoom()
    function = Function()
    # endregion

    # region 2. camera open
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    current_date_time = datetime.datetime.now().strftime('%Y%m%d0_%H%M%S')
    filename = f'{current_date_time}.avi'
    out = cv2.VideoWriter(filename, fourcc, 20.0, (1920, 960))
    step = 0
    class_name = None
    # endregion

    # region 3. initial ptz settings
    lat, lon = cube.get_pos()
    heading = cube.get_direction()
    pan, tilt, zoom = function.init_ptz_angle(lat, lon, Variable.drone_lat, Variable.drone_lon, Variable.drone_height,
                                              heading)
    pan += Variable.pan_compensate
    tilt += Variable.tilt_compensate

    ptz.yaw_pitch(pan, tilt, 50, 50)
    ptz.zoom(zoom)
    time.sleep(0.1)
    w_history = deque(maxlen=7)
    h_history = deque(maxlen=7)
    check_ptz_difference = True
    # endregion

    try:
        # First : handle ptz
        while True:
            if vision.frame is not None:
                # ptz.focus(-1)
                processed_frame, x, y, w, h = detect.process_frame(vision.frame.copy())
                screen_frame = cv2.resize(processed_frame, (960, 480))
                cv2.imshow('Processed RTSP Stream', screen_frame)

                ptz.get_angle()
                if keyboard.is_pressed('up'):
                    ptz.yaw_pitch(yaw=ptz.yaw, pitch=ptz.pitch + 1, yaw_speed=1, pitch_speed=10)
                if keyboard.is_pressed('down'):
                    ptz.yaw_pitch(yaw=ptz.yaw, pitch=ptz.pitch - 1, yaw_speed=1, pitch_speed=10)
                if keyboard.is_pressed('right'):
                    ptz.yaw_pitch(yaw=ptz.yaw + 1, pitch=ptz.pitch, yaw_speed=10, pitch_speed=1)
                if keyboard.is_pressed('left'):
                    ptz.yaw_pitch(yaw=ptz.yaw - 1, pitch=ptz.pitch, yaw_speed=10, pitch_speed=1)
                if keyboard.is_pressed('w'):
                    w, h = 1, 1
                    zoom = set_zoom.change_zoom(w, h, zoom)
                    ptz.zoom(zoom)
                if keyboard.is_pressed('s'):
                    w, h = 1000, 1000
                    zoom = set_zoom.change_zoom(w, h, zoom)
                    ptz.zoom(zoom)
                time.sleep(0.01)

                if keyboard.is_pressed('space'):
                    break
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        past_ptz_angle = ptz.get_angle()

        # Second : auto ptz
        while True:
            if vision.frame is not None:
                processed_frame, x, y, w, h = detect.process_frame(vision.frame.copy())
                cv2.putText(processed_frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                w_history.append(w)
                h_history.append(h)

                step += 1
                if step % 31 == 0:
                    class_name = recognize.bounding_box(processed_frame, x, y, w, h)

                if check_ptz_difference and step % 100 == 0:
                    current_ptz_angle = ptz.get_angle()
                    ptz_difference = np.array(current_ptz_angle) - np.array(past_ptz_angle)
                    print("pan_compensate =", ptz_difference[0])
                    print("tilt_compensate =", ptz_difference[1])
                    check_ptz_difference = False

                if step % 77 == 0:
                    w, h = sum(w_history) / len(w_history), sum(h_history) / len(h_history)
                    new_zoom = set_zoom.change_zoom(w, h, zoom)
                    if new_zoom == zoom:
                        continue
                    zoom = new_zoom
                    ptz.zoom(zoom)

                now = datetime.datetime.now()
                time_str = now.strftime("%H:%M:%S")
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(processed_frame, time_str, (10, 50), font, 1, (50, 50, 50), 2, cv2.LINE_AA)

                screen_frame = cv2.resize(processed_frame, (960, 480))

                cv2.imshow('Processed RTSP Stream', screen_frame)
                out.write(processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # region ptz control, speed setting
                dx = (x + w / 2) - 960
                dy = -(y + h / 2) + 480

                # memorize.add(dx, dy)
                # past_dataset = memorize.get_all()
                # if len(past_dataset) == 5:
                #     dx = lstm.predict(past_dataset)[0][0]
                #     dy = lstm.predict(past_dataset)[0][1]

                ptz.get_angle()
                move_x, vel_x = function.your_MS(dx, zoom)
                move_y, vel_y = function.your_MS(dy, zoom)

                ptz.yaw_pitch(yaw=ptz.yaw + move_x, pitch=ptz.pitch + move_y, yaw_speed=vel_x, pitch_speed=vel_y)
                # endregion
                function.save_ptz_data_to_csv(ptz.yaw, ptz.pitch, zoom, w, h)

    finally:
        out.release()
        cv2.destroyAllWindows()
