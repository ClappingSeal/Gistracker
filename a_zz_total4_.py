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

tf.debugging.set_log_device_placement(True)
logging.getLogger('dronekit').setLevel(logging.CRITICAL)


class Variable:
    drone_lat = 35.228512
    drone_lon = 226.840277
    drone_height = 80

    port_arduino = 'COM14'
    port_cube = 'COM5'
    ip_address = '192.168.1.3'


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
        time.sleep(0.1)

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
            return x, y, w, h
        else:
            return 960, 480, 0, 0


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
        self.zoom_array1 = [1, 3, 5, 8, 10, 13, 15, 20, 25, 30, 40, 50, 60, 61, 62, 63, 64]
        self.zoom_array2 = [1, 3, 5, 8, 10, 13, 15, 20, 25, 30, 40, 50, 60, 120, 150, 300, 600]
        self.expanding_area = 100
        self.reducing_area = 400

    def change_zoom(self, w, h, current_zoom):
        area = w * h
        if area < self.expanding_area:
            indices = np.where(np.array(self.zoom_array1) == current_zoom)[0]
            return self.zoom_array1[int(indices[0]) + 1]
        if area > self.reducing_area:
            indices = np.where(np.array(self.zoom_array1) == current_zoom)[0]
            return self.zoom_array1[int(indices[0]) - 1]


class Function:
    def your_speed(s, zoom, small=5):
        if zoom <= 19:
            divide = 1
        elif 19 < zoom <= 40:
            divide = 2
        elif 40 < zoom < 60:
            divide = 3
        else:
            divide = 5

        s = abs(s)
        if s > 150:
            return int(s / 20 / divide)
        else:
            return int(small / divide)

    def your_move(zoom):
        if zoom <= 20:
            return 30
        elif zoom <= 30:
            return 40
        else:
            return 70

    def save_ptz_data_to_csv(yaw, pitch, name="ptz_data.csv"):
        file_exists = os.path.isfile(name)
        with open(name, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Timestamp", "Yaw", "Pitch"])
            writer.writerow([datetime.now(), yaw, pitch])

    def init_ptz_angle(camera_lat, camera_lon, drone_lat, drone_lon, drone_h, camera_heading):
        delta_x = drone_lat - camera_lat
        delta_y = drone_lon - camera_lon
        d = math.sqrt(delta_x ** 2 + delta_y ** 2) * 6371000 * math.pi / 180

        pan = math.atan(delta_y / delta_x) * 180 / math.pi - camera_heading
        tilt = math.atan(drone_h / d) * 180 / math.pi

        return pan, tilt


if __name__ == "__main__":
    # region class declare
    ptz = PTZ()
    vision = VISION()
    cube = CubeOrange()
    detect = Detect1(vision)
    recognize = Recognize()
    lstm = LSTM('scaler.pkl', 'lstm_drone_positions_model.tflite')
    memorize = Memorize()
    set_zoom = SetZoom()
    # endregion

    lat, lon = cube.get_pos()
    heading = cube.get_direction()
    pan, tilt = Function.init_ptz_angle(lat, lon, Variable.drone_lat, Variable.drone_lon, Variable.drone_height,
                                        heading)

    ptz.yaw_pitch(pan, tilt, 50, 50)

    zoom = 30  # 추후 드론 거리를 기반으로 줌 하기
    ptz.zoom(zoom)
    time.sleep(5)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    current_date_time = datetime.now().strftime('%Y%m%d0_%H%M%S')
    filename = f'{current_date_time}.avi'
    out = cv2.VideoWriter(filename, fourcc, 20.0, (1920, 960))
    step = 0
    class_name = None

    try:
        while True:
            if vision.frame is not None:
                processed_frame, x, y, w, h = detect.process_frame(vision.frame.copy())
                # cv2.putText(processed_frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                step += 1
                if step % 31 == 0:
                    class_name = recognize.bounding_box(processed_frame, x, y, w, h)

                if step % 50 == 0:
                    zoom = set_zoom.change_zoom(2, 3, zoom)
                    ptz.zoom(zoom)

                screen_frame = cv2.resize(processed_frame, (960, 480))
                cv2.imshow('Processed RTSP Stream', screen_frame)
                out.write(processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # region ptz control, speed setting
                dx = (x + w / 2) - 960
                dy = -(y + h / 2) + 480
                print(w * h)

                memorize.add(dx, dy)
                past_dataset = memorize.get_all()
                if len(past_dataset) == 5:
                    dx = lstm.predict(past_dataset)[0][0]
                    dy = lstm.predict(past_dataset)[0][1]

                ptz.get_angle()
                ptz.yaw_pitch(yaw=ptz.yaw + dx / Function.your_move(zoom),
                              pitch=ptz.pitch + dy / Function.your_move(zoom),
                              yaw_speed=Function.your_speed(dx, zoom),
                              pitch_speed=Function.your_speed(dy, zoom))

                # endregion
                Function.save_ptz_data_to_csv(ptz.yaw, ptz.pitch)

    finally:
        out.release()
        cv2.destroyAllWindows()
