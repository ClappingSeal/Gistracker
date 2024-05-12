import cv2

# 마우스 콜백 함수
def draw_rectangle(event, x, y, flags, param):
    global x_start, y_start, x_end, y_end, drawing, track_window, tracker_initialized, tracker

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x_start, y_start = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            x_end, y_end = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x_end, y_end = x, y

        # 선택된 영역으로 트래커 초기화
        if not tracker_initialized:
            track_window = (x_start, y_start, x_end - x_start, y_end - y_start)
            tracker = cv2.TrackerKCF_create()
            tracker.init(frame, track_window)
            tracker_initialized = True

def main():
    global x_start, y_start, x_end, y_end, drawing, tracker_initialized, tracker, frame
    drawing = False
    tracker_initialized = False
    x_start = y_start = x_end = y_end = 0

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", draw_rectangle)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 트래커가 초기화된 경우, 트래킹 업데이트
        if tracker_initialized:
            success, track_window = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in track_window]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 마우스로 선택 영역을 그림 (트래커 초기화 전에만)
        if not tracker_initialized and drawing:
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
