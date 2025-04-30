import torch
import cv2
import sys
import logging
import os
from contextlib import ExitStack

from face_system import (
    FaceBank,
    FaceDetector,
    FaceMesh,
    FaceProcessor,
    FaceRecognizer,
    FaceTracker,
)

from utils import parse_args, FPSCounter, InfoOverlay, VideoWriter


def configure_logging(enable_logging, destination, log_file):
    if not enable_logging:
        return

    handlers = []
    if destination in ("file", "both"):
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        handlers.append(logging.FileHandler(log_file))

    if destination in ("terminal", "both"):
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def main():
    try:
        args = parse_args()
        configure_logging(args.log, args.log_destination, args.log_file)

        logging.info("Application started with arguments: %s", vars(args))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        amp_enabled = args.mode == "amp" and device.type == "cuda"
        logging.info("Using device: %s | AMP enabled: %s", device, amp_enabled)

        face_bank = FaceBank(face_bank_path=args.face_bank_path)
        logging.info("Loaded face bank from: %s", args.face_bank_path)

        face_detector = FaceDetector(
            model=args.face_detector,
            tracker_config=args.tracker_config,
            det_threshold=args.det_threshold,
            device=device,
        )
        face_recognizer = FaceRecognizer(
            model_name=args.face_recognizer, device=device, amp_enabled=amp_enabled
        )
        source = int(args.source) if args.source.isdigit() else args.source
        detection_result_gen = face_detector.get_result_generator(source)
        face_tracker = FaceTracker(recognize_threshold=args.rec_threshold)
        fps_counter = FPSCounter(smoothing=args.fps_smoothing)

        info_overlay = InfoOverlay()

        with ExitStack() as stack:
            video_writer = None
            face_mesh = stack.enter_context(FaceMesh())
            face_processor = FaceProcessor(
                face_mesh=face_mesh,
                min_face_size=args.min_face_size,
                det_threshold=args.det_threshold,
                blurry_threshold=args.blurry_threshold,
            )
            logging.info("Face processor initialized")

            for detection_result in detection_result_gen:
                fps_counter.update()
                face_tracker.increment_frame()

                frame = detection_result.orig_img

                if video_writer is None and args.save_video:
                    video_writer = stack.enter_context(
                        VideoWriter(
                            output_path=args.output_path,
                            frame_size=(frame.shape[1], frame.shape[0]),
                        )
                    )
                    logging.info("Initialized video writer: %s", args.output_path)

                faces_info = face_processor.process_frame(
                    frame=frame,
                    detection_result=detection_result,
                )

                if faces_info:
                    aligned_faces, track_ids, bounding_boxes = faces_info

                    if aligned_faces:
                        embeddings = face_recognizer.recognize_batch(aligned_faces)
                        for emb, tid, box in zip(embeddings, track_ids, bounding_boxes):
                            identity, sim = face_bank.match_face(
                                embedding=emb, threshold=args.rec_threshold
                            )

                            is_new_identity = (
                                tid not in face_tracker.id_memory
                                and sim >= args.rec_threshold
                            )

                            face_tracker.update(
                                track_id=tid, identity=identity, similarity=sim
                            )

                            if is_new_identity:
                                logging.info(
                                    "Track ID %d: Identity='%s', Similarity=%.4f",
                                    tid,
                                    identity,
                                    sim,
                                )

                            if face_tracker.get_age(tid) < args.min_track_age:
                                logging.debug(
                                    "Track %d age (%d) below min_track_age, skipping overlay",
                                    tid,
                                    face_tracker.get_age(tid),
                                )
                                continue

                            persistent_identity = face_tracker.get_identity(
                                track_id=tid
                            )
                            info_overlay.overlay_identity(
                                frame=frame,
                                track_id=tid,
                                identity=persistent_identity,
                                similarity=sim,
                                box=box,
                            )
                if args.show_fps:
                    info_overlay.overlay_fps(frame=frame, fps=fps_counter.get_fps())
                    logging.debug("FPS: %.2f", fps_counter.get_fps())

                if args.save_video and video_writer is not None:
                    video_writer.write(frame)

                cv2.imshow("Face Detection + Recognition", frame)

                face_tracker.cleanup(max_inactive_frames=args.max_inactive_frames)

                if cv2.waitKey(1) == ord("q"):
                    break

            cv2.destroyAllWindows()
            logging.info("Application terminated gracefully")
            sys.exit(0)

    except Exception:
        logging.exception("An unexpected error occured in the main function.")
        sys.exit(1)


if __name__ == "__main__":
    main()
