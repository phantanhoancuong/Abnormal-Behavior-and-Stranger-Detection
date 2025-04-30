import cv2


class VideoWriter:
    def __init__(self, output_path, frame_size, fps=30):
        self.output_path = output_path
        self.frame_size = frame_size
        self.fps = fps

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            filename=self.output_path,
            fourcc=fourcc,
            fps=self.fps,
            frameSize=self.frame_size,
        )

    def write(self, frame):
        if self.writer is not None:
            self.writer.write(frame)

    def release(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, trackback):
        self.release()
