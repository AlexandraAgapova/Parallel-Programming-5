import argparse
import cv2
import multiprocessing as mp
import time
from ultralytics import YOLO


class YOLOPoseProcessor:
    def __init__(self, model_path="yolov8s-pose.pt"):
        self.model = YOLO(model_path)

    def __del__(self):
        del self.model

    def process(self, frame):
        result = self.model.predict(source=frame, verbose=False, device="cpu")[0]
        return result.plot()


class VideoCaptureRALL:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError("Cannot open video")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_all_frames(self):
        frames = []
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frames.append(frame)
        return frames

    def __del__(self):
        self.cap.release()


class VideoWriterRALL:
    def __init__(self, path, fps, width, height):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(path, fourcc, fps, (width, height))

    def write_frames(self, frames):
        for f in frames:
            self.out.write(f)

    def __del__(self):
        self.out.release()


def process_single(frames):
    model = YOLOPoseProcessor()
    return [model.process(frame) for frame in frames]


def worker_process(frame_data):
    model = YOLOPoseProcessor()
    idx, frame = frame_data
    processed = model.process(frame)
    return idx, processed 


def process_multi(frames, num_processes):
    with mp.Pool(processes=num_processes) as pool:
        indexed_frames = list(enumerate(frames))
        results = pool.map(worker_process, indexed_frames)
        results.sort(key=lambda x: x[0]) 
        return [frame for _, frame in results]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Путь к входному видео (разрешение 640x480)")
    parser.add_argument("--output", required=True, help="Имя выходного видеофайла")
    parser.add_argument("--mode", choices=["single", "multi"], default="single", help="Режим выполнения")
    parser.add_argument("--procs", type=int, default=4, help="Число процессов (только для multi)")
    args = parser.parse_args()

    reader = VideoCaptureRALL(args.input)
    writer = VideoWriterRALL(args.output, reader.fps, reader.width, reader.height)
    frames = reader.read_all_frames()

    start = time.time()
    if args.mode == "single":
        processed = process_single(frames)
        print(f"[Single-thread] Elapsed time: {time.time() - start:.2f} sec")
    else:
        processed = process_multi(frames, args.procs)
        print(f"[Multi-process ({args.procs})] Elapsed time: {time.time() - start:.2f} sec")

    writer.write_frames(processed)


if __name__ == "__main__":
    mp.freeze_support() 
    main()
