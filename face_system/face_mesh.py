import mediapipe as mp


class FaceMesh:
    def __init__(self, static_image_mode=False, refine_landmarks=True):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode, refine_landmarks=refine_landmarks
        )

    def process(self, image_rgb):
        return self.face_mesh.process(image_rgb)

    def close(self):
        self.face_mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
