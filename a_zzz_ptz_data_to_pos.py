import csv
import math
import pandas as pd
import matplotlib.pyplot as plt
import glob
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from model import Model
import os

Mission_num = 3


class YourAbsolutePos:
    def __init__(self, file_path):
        self.drone_init_lat = 35.227188
        self.drone_init_lon = 126.839626
        self.camera_lat = 35.2267883
        self.camera_lon = 126.8407577

        self.file_path = file_path

        with open(self.file_path, 'r') as file:
            next(file)
            second_line = next(file)
            data = second_line.strip().split('\t')
            values = data[0].split(',')
        self.init_theta = float(values[1])
        self.init_zoom = float(values[3])
        self.init_w = float(values[4])

    def calculate(self):
        new_data = []

        with open(self.file_path, 'r', newline='') as file:
            reader = csv.reader(file)
            next(reader)

            for line in reader:
                if len(line) < 6:
                    continue
                time = line[0]
                yaw = float(line[1])
                pitch = float(line[2])
                zoom = float(line[3])
                w = float(line[4])

                init_r = math.hypot(self.drone_init_lat - self.camera_lat,
                                    (self.drone_init_lon - self.camera_lon) * math.cos(
                                        math.radians(self.camera_lat))) * 6371000 * math.pi / 180

                if w == 0:
                    x = 0
                    y = 0
                    z = 0
                else:
                    x = (self.init_w / self.init_zoom * init_r * math.sin(math.radians(yaw)) * zoom / w)
                    y = (self.init_w / self.init_zoom * init_r * math.cos(math.radians(yaw)) * zoom / w)
                    z = math.sqrt(x ** 2 + y ** 2) * math.tan(pitch * math.pi / 180)

                lat = x / 6371000 + self.camera_lat
                lon = y / 6371000 + self.camera_lon

                new_data.append([time, x, y, z, lat, lon])

        new_file_path = f'{Mission_num}/drone_coordinates.csv'
        with open(new_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'x', 'y', 'z', 'lat', 'lon'])
            writer.writerows(new_data)

        print(f"Data saved to {new_file_path}")


class Draw:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(self.file_path)
        self.sampled_data = self.data.iloc[::5]

    def gps_info(self):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.sampled_data['x'], self.sampled_data['y'], alpha=0.5)

        for i, point in self.sampled_data.iterrows():
            plt.text(point['x'], point['y'], ' ' + point['Timestamp'], fontsize=6, color='gray')

        plt.title('Plot x, y GPS info')
        plt.xlabel('x Coordinate')
        plt.ylabel('y Coordinate')
        plt.grid(True)
        save_path = f'{Mission_num}/Plot x, y GPS info.png'
        plt.savefig(save_path)
        plt.show()


class NumRecognize:
    def __init__(self):
        # 모델 초기화
        self.id_model = YOLO('best.pt')
        self.svhn_model = Model()
        self.svhn_model.restore('model-193000.pth')
        self.svhn_model.cuda()
        self.transform = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.CenterCrop([54, 54]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def read_id(self, image_path):
        image = cv2.imread(image_path)
        results = self.id_model(image, verbose=False)
        for r in results:
            if r.obb.xyxyxyxy.shape[0] > 0:  # 비어 있지 않은지 확인
                a = r.obb.xyxyxyxy[0].tolist()
                pts_src = np.array(a, np.float32)
                width = max(np.linalg.norm(pts_src[0] - pts_src[1]), np.linalg.norm(pts_src[2] - pts_src[3]))
                height = max(np.linalg.norm(pts_src[0] - pts_src[3]), np.linalg.norm(pts_src[1] - pts_src[2]))

                pts_dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
                M = cv2.getPerspectiveTransform(pts_src, pts_dst)
                warped_image = cv2.warpPerspective(image, M, (int(width), int(height)))
                warped_image = cv2.transpose(warped_image)
                warped_image = cv2.flip(warped_image, -1)
                img = cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(img)
                id = 0
                dec = 1
                with torch.no_grad():
                    image = im_pil.convert('RGB')
                    image = self.transform(image)
                    images = image.unsqueeze(dim=0).cuda()
                    length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits = self.svhn_model.eval()(
                        images)
                    digit_predictions = [logit.max(1)[1].item() for logit in
                                         [digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits]]
                    for i in reversed(digit_predictions):
                        if i != 10:  # 10은 빈 값
                            id += i * dec
                            dec *= 10
                    return id
        # 유효한 결과가 없을 경우
        print("No valid results found.")
        return None

    def write_num(self, mission_folder):
        # 파일 찾기
        pattern = os.path.join(mission_folder, 'cropped*.png')
        files = glob.glob(pattern)
        if files:
            largest_file = max(files, key=os.path.getsize)
        else:
            print("No files found matching the pattern.")
            return

        # ID 추출
        id = self.read_id(largest_file)
        if id is None:
            print("Failed to extract ID from image.")
            return

        # ID를 이미지에 추가
        image = Image.open(largest_file)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("arial.ttf", 40)  # 글꼴과 크기 선택
        draw.text((10, 10), str(id), fill="red", font=font)  # 위치와 색상 선택

        # 새로운 이름으로 이미지 저장
        new_filename = os.path.join(mission_folder, 'Get_num.png')
        image.save(new_filename)

        print(f"Processed image saved as {new_filename}")


def main():
    file_path1 = f'{Mission_num}/ptz_data.csv'
    file_path2 = f'{Mission_num}'
    file_path3 = f'{Mission_num}/drone_coordinates.csv'
    yps = YourAbsolutePos(file_path1)
    yps.calculate()
    num = NumRecognize()
    num.write_num(file_path2)
    draw = Draw(file_path3)
    draw.gps_info()


if __name__ == "__main__":
    main()
