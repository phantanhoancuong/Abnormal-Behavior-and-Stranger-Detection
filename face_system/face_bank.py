import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class FaceBank:
    def __init__(self, face_bank_path):
        self.face_bank_path = face_bank_path
        self.face_bank = self._load_face_bank()

    def _load_face_bank(self):
        try:
            with open(self.face_bank_path, "rb") as f:
                face_bank = pickle.load(f)
            print(f"[INFO] Loaded {len(face_bank)} identities from face bank.")
            return face_bank
        except Exception as e:
            print(f"[ERROR] Failed to load face bank: {e}")
            return {}

    def save(self):
        with open(self.face_bank_path, "wb") as f:
            pickle.dump(self.face_bank, f)
        print(f"[INFO] Face bank saved at {self.face_bank_path}.")

    def match_face(self, embedding, threshold=0.5):
        max_sim = -1
        identity = "Stranger"

        for name, emb_list in self.face_bank.items():
            sims = cosine_similarity([embedding], emb_list)[0]
            best_sim = np.max(sims)
            if best_sim > max_sim:
                max_sim = best_sim
                identity = name

        if max_sim < threshold:
            identity = "Stranger"

        return identity, max_sim
