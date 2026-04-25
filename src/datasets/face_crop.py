from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from src.utils.config import load_config, resolve_path
from src.utils.runtime import append_log


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float


class FaceDetector:
    def __init__(self) -> None:
        self.backend = None
        self.detector = None
        self._init_backend()

    def _init_backend(self) -> None:
        try:
            from retinaface import RetinaFace  # type: ignore

            self.backend = "retinaface"
            self.detector = RetinaFace
            return
        except ImportError:
            pass

        try:
            from facenet_pytorch import MTCNN  # type: ignore

            self.backend = "mtcnn"
            self.detector = MTCNN(keep_all=True, device="cpu")
            return
        except ImportError:
            pass

        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.backend = "haar"
        self.detector = cascade

    def detect(self, image_bgr) -> list[Detection]:
        if self.backend == "retinaface":
            detections = self.detector.detect_faces(image_bgr)
            boxes: list[Detection] = []
            if isinstance(detections, dict):
                for item in detections.values():
                    x1, y1, x2, y2 = item["facial_area"]
                    boxes.append(Detection(x1, y1, x2, y2, item.get("score", 0.0)))
            return boxes

        if self.backend == "mtcnn":
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            boxes, probs = self.detector.detect(image_rgb)
            outputs: list[Detection] = []
            if boxes is None:
                return outputs
            for box, score in zip(boxes, probs):
                x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                outputs.append(Detection(x1, y1, x2, y2, float(score)))
            return outputs

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        return [Detection(int(x), int(y), int(x + w), int(y + h), 1.0) for (x, y, w, h) in detections]


def expand_box(box: Detection, width: int, height: int, margin: float = 0.25) -> tuple[int, int, int, int]:
    box_w = box.x2 - box.x1
    box_h = box.y2 - box.y1
    dx = int(box_w * margin)
    dy = int(box_h * margin)
    return (
        max(box.x1 - dx, 0),
        max(box.y1 - dy, 0),
        min(box.x2 + dx, width),
        min(box.y2 + dy, height),
    )


def select_primary_face(detections: list[Detection]) -> Detection | None:
    if not detections:
        return None
    return sorted(detections, key=lambda det: ((det.x2 - det.x1) * (det.y2 - det.y1), det.score), reverse=True)[0]


def crop_faces(
    frame_df: pd.DataFrame,
    output_root: Path,
    image_size: int,
    min_valid_frames: int,
    log_path: Path,
) -> pd.DataFrame:
    detector = FaceDetector()
    rows: list[dict[str, object]] = []

    for video_id, group in tqdm(frame_df.groupby("video_id"), desc=f"Cropping faces -> {output_root.name}"):
        video_rows: list[dict[str, object]] = []
        saved_paths: list[Path] = []
        for row in group.to_dict("records"):
            image = cv2.imread(str(row["frame_path"]))
            if image is None:
                append_log(f"[face_crop] failed read {row['frame_path']}", log_path)
                continue
            detections = detector.detect(image)
            face = select_primary_face(detections)
            if face is None:
                append_log(f"[face_crop] no face detected {row['frame_path']}", log_path)
                continue
            height, width = image.shape[:2]
            x1, y1, x2, y2 = expand_box(face, width=width, height=height)
            crop = image[y1:y2, x1:x2]
            crop = cv2.resize(crop, (image_size, image_size))
            label_name = "real" if int(row["label"]) == 0 else "fake"
            output_path = output_root / row["split"] / label_name / row["video_id"] / f"{row['frame_id']}.jpg"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), crop)
            saved_paths.append(output_path)
            video_rows.append(
                {
                    "video_id": row["video_id"],
                    "frame_id": row["frame_id"],
                    "face_path": str(output_path.resolve()),
                    "label": int(row["label"]),
                    "split": row["split"],
                    "dataset": row["dataset"],
                }
            )

        if len(video_rows) < min_valid_frames:
            append_log(f"[face_crop] dropped {video_id}: valid_frames={len(video_rows)}", log_path)
            for path in saved_paths:
                if path.exists():
                    path.unlink()
            continue
        rows.extend(video_rows)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect faces and crop frame manifests.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--frame-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--log-name", default="face_crop.log")
    args = parser.parse_args()

    config = load_config(args.config)
    frame_df = pd.read_csv(resolve_path(args.frame_manifest))
    output_root = resolve_path(args.output_root)
    log_path = resolve_path(config["paths"]["log_dir"]) / args.log_name

    manifest = crop_faces(
        frame_df=frame_df,
        output_root=output_root,
        image_size=config["data"]["image_size"],
        min_valid_frames=config["data"]["min_valid_frames"],
        log_path=log_path,
    )
    manifest_out = resolve_path(args.manifest_out)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_out, index=False)
    print(f"Saved face manifest to {manifest_out}")


if __name__ == "__main__":
    main()
