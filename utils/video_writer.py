import cv2
import logging
import os


class VideoWriter:
    def __init__(self, output_path, frame_size, fps=30):
        self.output_path = output_path
        self.frame_size = frame_size  # (width, height)
        self.fps = fps

        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                logging.info(f"Created output directory: {output_dir}")
        else:
            logging.warning(
                "Output path has no directory; saving video to current working directory"
            )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            filename=self.output_path,
            fourcc=fourcc,
            fps=self.fps,
            frameSize=self.frame_size,
        )
        logging.info(
            f"VideoWriter initialized: {self.output_path} ({self.frame_size[0]}x{self.frame_size[1]} @ {self.fps} FPS)"
        )

    def write(self, frame):
        if self.writer is not None:
            self.writer.write(frame)

    def release(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            logging.info("VideoWriter released")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type and exc_type is not SystemExit:
            logging.error(
                "Exception occurred in VideoWriter context manager",
                exc_info=(exc_type, exc_value, traceback),
            )
        self.release()
