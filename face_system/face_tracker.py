class FaceTracker:
    def __init__(self, recognize_threshold):
        self.recognize_threshold = recognize_threshold
        self.id_memory = {}
        self.age_memory = {}
        self.last_seen_frame = {}
        self.global_frame_count = 0

    def update(self, track_id, identity, similarity):
        if track_id is None:
            return

        if track_id not in self.age_memory:
            self.age_memory[track_id] = 1
        else:
            self.age_memory[track_id] += 1

        self.last_seen_frame[track_id] = self.global_frame_count

        if track_id not in self.id_memory:
            if similarity >= self.recognize_threshold:
                self.id_memory[track_id] = {
                    "identity": identity,
                    "similarity": similarity,
                }
        else:
            if similarity > self.id_memory[track_id]["similarity"]:
                self.id_memory[track_id] = {
                    "identity": identity,
                    "similarity": similarity,
                }

    def get_identity(self, track_id):
        if track_id is None:
            return "Unknown"
        return self.id_memory.get(track_id, {"identity": "Unknown"})["identity"]

    def get_age(self, track_id):
        return self.age_memory.get(track_id, 0)

    def increment_frame(self):
        self.global_frame_count += 1

    def cleanup(self, max_inactive_frames):
        to_delete = []
        for track_id, last_seen in self.last_seen_frame.items():
            if (self.global_frame_count - last_seen) > max_inactive_frames:
                to_delete.append(track_id)

        for track_id in to_delete:
            self.id_memory.pop(track_id, None)
            self.age_memory.pop(track_id, None)
            self.last_seen_frame.pop(track_id, None)

    def reset(self):
        self.id_memory.clear()
        self.age_memory.clear()
        self.last_seen_frame.clear()
        self.global_frame_count = 0
