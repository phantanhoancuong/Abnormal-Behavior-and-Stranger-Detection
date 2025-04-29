import cv2


class InfoOverlay:
    def overlay_fps(self, frame, fps):
        text = f"FPS: {fps:.1f}"
        cv2.putText(
            frame,
            text,
            (20, 40),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=(255, 255, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    def overlay_identity(self, frame, track_id, identity, similarity, box):
        x1, y1, x2, y2 = box
        color = (0, 0, 255) if identity == "Stranger" else (0, 255, 0)
        label = f"ID: {track_id} | {identity} | Sim: {similarity:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)

        (text_width, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        center_x = (x1 + x2) // 2
        text_x = center_x - (text_width // 2)
        text_y = max(y1 - 10, 0)

        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )
