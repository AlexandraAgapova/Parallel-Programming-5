import cv2
import time
from ultralytics import YOLO

class YOLOPoseProcessor:
    def __init__(self, model_path="yolov8s-pose.pt"):
        self.model = YOLO(model_path)

    def process(self, frame):
        result = self.model.predict(source=frame, verbose=False, device="cpu")[0]
        return result.plot()

class CameraCaptureRALL:
    def __init__(self, cam_index=0):
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise IOError("Cannot open camera")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def get_frame(self):
        ret, frame = self.cap.read()
        return ret, frame

    def __del__(self):
        self.cap.release()


def main():
    cap = CameraCaptureRALL()
    model = YOLOPoseProcessor()

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.get_frame()
        if not ret:
            break

        processed = model.process(frame)
        cv2.imshow("RealTime", processed)

        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"FPS: {fps:.2f}", end='\r')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"\nAverage FPS: {frame_count / (time.time() - start_time):.2f}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
