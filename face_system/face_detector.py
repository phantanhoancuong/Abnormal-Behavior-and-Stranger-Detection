from ultralytics import YOLO


class FaceDetector:
    def __init__(self, model, tracker_config, det_threshold, device):
        self.model = YOLO(model, verbose=False)
        self.tracker_config = tracker_config
        self.det_threshold = det_threshold
        self.device = device

    def get_result_generator(self, source):
        detection_results = self.model.track(
            source=source,
            persist=True,
            stream=True,
            tracker=self.tracker_config,
            verbose=False,
        )

        return detection_results
