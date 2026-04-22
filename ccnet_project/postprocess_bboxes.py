import argparse
import csv
import json
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
except ImportError:
    cv2 = None


BoxRecord = Dict[str, float | int | str]
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def read_grayscale(path: Path) -> np.ndarray:
    image = np.array(Image.open(path).convert("L"))
    assert image.ndim == 2, f"Expected grayscale image, got shape {image.shape}: {path}"
    return image


def connected_components(mask: np.ndarray) -> List[np.ndarray]:
    assert mask.ndim == 2, f"Mask must be [H,W], got: {mask.shape}"
    visited = np.zeros(mask.shape, dtype=bool)
    components: List[np.ndarray] = []
    height, width = mask.shape

    ys, xs = np.nonzero(mask)
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if visited[start_y, start_x]:
            continue

        queue: deque[Tuple[int, int]] = deque([(start_y, start_x)])
        visited[start_y, start_x] = True
        pixels: List[Tuple[int, int]] = []

        while queue:
            y, x = queue.popleft()
            pixels.append((y, x))

            for ny in (y - 1, y, y + 1):
                for nx in (x - 1, x, x + 1):
                    if ny == y and nx == x:
                        continue
                    if ny < 0 or nx < 0 or ny >= height or nx >= width:
                        continue
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    queue.append((ny, nx))

        components.append(np.array(pixels, dtype=np.int32))

    return components


def mask_to_boxes_cv2(
    binary: np.ndarray,
    prob: np.ndarray | None,
    min_area: int,
    min_width: int,
    min_height: int,
) -> List[BoxRecord]:
    assert cv2 is not None
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    boxes: List[BoxRecord] = []

    for label_id in range(1, num_labels):
        x_min = int(stats[label_id, cv2.CC_STAT_LEFT])
        y_min = int(stats[label_id, cv2.CC_STAT_TOP])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_id, cv2.CC_STAT_AREA])

        if area < min_area or width < min_width or height < min_height:
            continue

        record: BoxRecord = {
            "component_id": label_id,
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_min + width - 1,
            "y_max": y_min + height - 1,
            "width": width,
            "height": height,
            "area": area,
        }
        if prob is not None:
            component_probs = prob[labels == label_id].astype(np.float32) / 255.0
            record["score"] = float(component_probs.mean())
            record["max_score"] = float(component_probs.max())
        boxes.append(record)

    return boxes


def mask_to_boxes(
    pred: np.ndarray,
    prob: np.ndarray | None = None,
    threshold: int = 127,
    min_area: int = 10,
    min_width: int = 1,
    min_height: int = 1,
) -> List[BoxRecord]:
    binary = pred > threshold
    if cv2 is not None:
        return mask_to_boxes_cv2(binary, prob, min_area, min_width, min_height)

    components = connected_components(binary)
    boxes: List[BoxRecord] = []

    for component_id, pixels in enumerate(components, start=1):
        ys = pixels[:, 0]
        xs = pixels[:, 1]
        x_min = int(xs.min())
        y_min = int(ys.min())
        x_max = int(xs.max())
        y_max = int(ys.max())
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        area = int(pixels.shape[0])

        if area < min_area or width < min_width or height < min_height:
            continue

        record: BoxRecord = {
            "component_id": component_id,
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "width": width,
            "height": height,
            "area": area,
        }
        if prob is not None:
            component_probs = prob[ys, xs].astype(np.float32) / 255.0
            record["score"] = float(component_probs.mean())
            record["max_score"] = float(component_probs.max())
        boxes.append(record)

    return boxes


def process_prediction_file(
    pred_path: Path,
    prob_dir: Path | None,
    threshold: int,
    min_area: int,
    min_width: int,
    min_height: int,
) -> List[BoxRecord]:
    pred = read_grayscale(pred_path)
    prob = None
    if prob_dir is not None:
        prob_path = prob_dir / pred_path.name
        assert prob_path.exists(), f"Missing probability map for {pred_path.name}: {prob_path}"
        prob = read_grayscale(prob_path)
        assert prob.shape == pred.shape, f"prob/pred size mismatch: {prob_path} vs {pred_path}"

    boxes = mask_to_boxes(
        pred=pred,
        prob=prob,
        threshold=threshold,
        min_area=min_area,
        min_width=min_width,
        min_height=min_height,
    )
    for box in boxes:
        box["image"] = pred_path.name
        box["stem"] = pred_path.stem
    return boxes


def find_image_by_stem(image_dir: Path, stem: str) -> Path:
    matches = [
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.stem == stem and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    assert matches, f"Missing source image for stem '{stem}' in {image_dir}"
    assert len(matches) == 1, f"Multiple source images found for stem '{stem}': {matches}"
    return matches[0]


def scale_box_to_image(box: BoxRecord, pred_size: Tuple[int, int], image_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    pred_width, pred_height = pred_size
    image_width, image_height = image_size
    scale_x = image_width / pred_width
    scale_y = image_height / pred_height
    x_min = int(round(float(box["x_min"]) * scale_x))
    y_min = int(round(float(box["y_min"]) * scale_y))
    x_max = int(round((float(box["x_max"]) + 1) * scale_x)) - 1
    y_max = int(round((float(box["y_max"]) + 1) * scale_y)) - 1
    x_min = max(0, min(image_width - 1, x_min))
    y_min = max(0, min(image_height - 1, y_min))
    x_max = max(0, min(image_width - 1, x_max))
    y_max = max(0, min(image_height - 1, y_max))
    return x_min, y_min, x_max, y_max


def draw_boxes_on_image(
    image_path: Path,
    pred_shape: Tuple[int, int],
    boxes: List[BoxRecord],
    save_path: Path,
    line_width: int = 3,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    pred_height, pred_width = pred_shape

    for box in boxes:
        x_min, y_min, x_max, y_max = scale_box_to_image(box, (pred_width, pred_height), image.size)
        draw.rectangle((x_min, y_min, x_max, y_max), outline=(255, 0, 0), width=line_width)

        label = f"{int(box['component_id'])}"
        if "score" in box:
            label += f" {float(box['score']):.2f}"
        text_bbox = draw.textbbox((x_min, y_min), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_y = max(0, y_min - text_height - 4)
        draw.rectangle((x_min, text_y, x_min + text_width + 4, text_y + text_height + 4), fill=(255, 0, 0))
        draw.text((x_min + 2, text_y + 2), label, fill=(255, 255, 255), font=font)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(save_path)


def save_csv(records: List[BoxRecord], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "stem",
        "component_id",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "width",
        "height",
        "area",
        "score",
        "max_score",
    ]
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def save_json(records: List[BoxRecord], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: Dict[str, List[BoxRecord]] = {}
    for record in records:
        grouped.setdefault(str(record["image"]), []).append(record)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(grouped, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert pred/prob change maps to bounding boxes.")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing binary prediction maps.")
    parser.add_argument("--prob_dir", type=str, default=None, help="Directory containing probability maps with matching names.")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory containing original large images.")
    parser.add_argument("--save_vis_dir", type=str, default=None, help="Directory for images with drawn bounding boxes.")
    parser.add_argument("--save_csv", type=str, default="outputs/infer/bboxes.csv")
    parser.add_argument("--save_json", type=str, default="outputs/infer/bboxes.json")
    parser.add_argument("--threshold", type=int, default=127, help="Pixel threshold for binary pred maps in [0,255].")
    parser.add_argument("--min_area", type=int, default=10, help="Drop connected components smaller than this pixel area.")
    parser.add_argument("--min_width", type=int, default=1)
    parser.add_argument("--min_height", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    prob_dir = Path(args.prob_dir) if args.prob_dir else None
    image_dir = Path(args.image_dir) if args.image_dir else None
    save_vis_dir = Path(args.save_vis_dir) if args.save_vis_dir else None
    assert pred_dir.exists(), f"Missing pred_dir: {pred_dir}"
    if prob_dir is not None:
        assert prob_dir.exists(), f"Missing prob_dir: {prob_dir}"
    if image_dir is not None:
        assert image_dir.exists(), f"Missing image_dir: {image_dir}"
    if save_vis_dir is not None:
        assert image_dir is not None, "--save_vis_dir requires --image_dir"

    records: List[BoxRecord] = []
    pred_paths = sorted(path for path in pred_dir.iterdir() if path.is_file())
    assert pred_paths, f"No prediction files found in {pred_dir}"

    for pred_path in pred_paths:
        boxes = process_prediction_file(
            pred_path=pred_path,
            prob_dir=prob_dir,
            threshold=int(args.threshold),
            min_area=int(args.min_area),
            min_width=int(args.min_width),
            min_height=int(args.min_height),
        )
        records.extend(boxes)

        if image_dir is not None and save_vis_dir is not None:
            image_path = find_image_by_stem(image_dir, pred_path.stem)
            pred_shape = read_grayscale(pred_path).shape
            draw_boxes_on_image(
                image_path=image_path,
                pred_shape=pred_shape,
                boxes=boxes,
                save_path=save_vis_dir / f"{pred_path.stem}_boxes.png",
            )

    save_csv(records, Path(args.save_csv))
    save_json(records, Path(args.save_json))
    print(f"Saved {len(records)} boxes to {args.save_csv} and {args.save_json}")


if __name__ == "__main__":
    main()
