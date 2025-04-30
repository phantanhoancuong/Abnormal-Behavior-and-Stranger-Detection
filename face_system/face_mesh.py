import mediapipe as mp
import logging


class FaceMesh:
    def __init__(self, static_image_mode=False, refine_landmarks=True):
        logging.info(
            "Initializing FaceMesh (static_image_mode=%s, refine_landmarks=%s)",
            static_image_mode,
            refine_landmarks,
        )
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode, refine_landmarks=refine_landmarks
        )

    def process(self, image_rgb):
        return self.face_mesh.process(image_rgb)

    def close(self):
        self.face_mesh.close()
        logging.info("FaceMesh released.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            logging.error(
                "Exception occurred in FaceMesh context manager",
                exc_info=(exc_type, exc_value, traceback),
            )
        self.close()
