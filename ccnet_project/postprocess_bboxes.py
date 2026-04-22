import argparse
import csv
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import tifffile
except ImportError:
    tifffile = None

try:
    import geopandas as gpd
    from shapely.geometry import Polygon
except ImportError:
    gpd = None
    Polygon = None

try:
    from skimage import measure as sk_measure
except ImportError:
    sk_measure = None


BoxRecord = Dict[str, float | int | str]
GeoJsonFeature = Dict[str, object]
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
GEOTIFF_SUFFIXES = (".tif", ".tiff")


@dataclass(frozen=True)
class GeoTransform:
    origin_x: float
    origin_y: float
    pixel_width: float
    pixel_height: float
    tie_col: float = 0.0
    tie_row: float = 0.0
    matrix: Tuple[float, ...] | None = None

    def pixel_to_map(self, col: float, row: float) -> Tuple[float, float]:
        if self.matrix is not None:
            m = self.matrix
            x = m[0] * col + m[1] * row + m[3]
            y = m[4] * col + m[5] * row + m[7]
            return float(x), float(y)

        x = self.origin_x + (col - self.tie_col) * self.pixel_width
        y = self.origin_y - (row - self.tie_row) * self.pixel_height
        return float(x), float(y)


def read_grayscale(path: Path) -> np.ndarray:
    image = np.array(Image.open(path).convert("L"))
    assert image.ndim == 2, f"Expected grayscale image, got shape {image.shape}: {path}"
    return image


def read_geotiff_metadata(path: Path) -> Tuple[GeoTransform, str | None, Tuple[int, int]]:
    if tifffile is None:
        return read_geotiff_metadata_with_pil(path)

    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        tags = page.tags
        image_height = int(page.imagelength)
        image_width = int(page.imagewidth)

        matrix_tag = tags.get("ModelTransformationTag")
        if matrix_tag is not None:
            matrix = tuple(float(value) for value in matrix_tag.value)
            assert len(matrix) == 16, f"Invalid ModelTransformationTag in {path}"
            transform = GeoTransform(0.0, 0.0, 1.0, 1.0, matrix=matrix)
        else:
            scale_tag = tags.get("ModelPixelScaleTag")
            tiepoint_tag = tags.get("ModelTiepointTag")
            assert scale_tag is not None, f"Missing ModelPixelScaleTag in GeoTIFF: {path}"
            assert tiepoint_tag is not None, f"Missing ModelTiepointTag in GeoTIFF: {path}"

            scale = tuple(float(value) for value in scale_tag.value)
            tiepoint = tuple(float(value) for value in tiepoint_tag.value)
            assert len(scale) >= 2, f"Invalid ModelPixelScaleTag in {path}"
            assert len(tiepoint) >= 6, f"Invalid ModelTiepointTag in {path}"

            transform = GeoTransform(
                origin_x=tiepoint[3],
                origin_y=tiepoint[4],
                pixel_width=abs(scale[0]),
                pixel_height=abs(scale[1]),
                tie_col=tiepoint[0],
                tie_row=tiepoint[1],
            )

        epsg = read_epsg_from_geokeys(tags)
        return transform, epsg, (image_height, image_width)


def read_geotiff_metadata_with_pil(path: Path) -> Tuple[GeoTransform, str | None, Tuple[int, int]]:
    with Image.open(path) as image:
        tags = image.tag_v2
        image_width, image_height = image.size

        matrix_tag = tags.get(34264)
        if matrix_tag is not None:
            matrix = tuple(float(value) for value in matrix_tag)
            assert len(matrix) == 16, f"Invalid ModelTransformationTag in {path}"
            transform = GeoTransform(0.0, 0.0, 1.0, 1.0, matrix=matrix)
        else:
            scale_tag = tags.get(33550)
            tiepoint_tag = tags.get(33922)
            assert scale_tag is not None, f"Missing ModelPixelScaleTag in GeoTIFF: {path}"
            assert tiepoint_tag is not None, f"Missing ModelTiepointTag in GeoTIFF: {path}"

            scale = tuple(float(value) for value in scale_tag)
            tiepoint = tuple(float(value) for value in tiepoint_tag)
            assert len(scale) >= 2, f"Invalid ModelPixelScaleTag in {path}"
            assert len(tiepoint) >= 6, f"Invalid ModelTiepointTag in {path}"

            transform = GeoTransform(
                origin_x=tiepoint[3],
                origin_y=tiepoint[4],
                pixel_width=abs(scale[0]),
                pixel_height=abs(scale[1]),
                tie_col=tiepoint[0],
                tie_row=tiepoint[1],
            )

        epsg = read_epsg_from_pil_geokeys(tags)
        return transform, epsg, (image_height, image_width)


def read_epsg_from_geokeys(tags) -> str | None:
    geokey_tag = tags.get("GeoKeyDirectoryTag")
    if geokey_tag is None:
        return None

    values = tuple(int(value) for value in geokey_tag.value)
    if len(values) < 4:
        return None

    key_count = values[3]
    entries = values[4 : 4 + key_count * 4]
    epsg_keys = {3072, 2048}
    for offset in range(0, len(entries), 4):
        key_id, tiff_tag_location, count, value_offset = entries[offset : offset + 4]
        if key_id in epsg_keys and tiff_tag_location == 0 and count == 1 and value_offset > 0:
            return f"EPSG:{value_offset}"
    return None


def read_epsg_from_pil_geokeys(tags) -> str | None:
    geokey_tag = tags.get(34735)
    if geokey_tag is None:
        return None

    values = tuple(int(value) for value in geokey_tag)
    if len(values) < 4:
        return None

    key_count = values[3]
    entries = values[4 : 4 + key_count * 4]
    epsg_keys = {3072, 2048}
    for offset in range(0, len(entries), 4):
        key_id, tiff_tag_location, count, value_offset = entries[offset : offset + 4]
        if key_id in epsg_keys and tiff_tag_location == 0 and count == 1 and value_offset > 0:
            return f"EPSG:{value_offset}"
    return None


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


def find_geotiff_by_stem(tile_image_dir: Path, stem: str) -> Path:
    matches = [
        path
        for path in tile_image_dir.iterdir()
        if path.is_file() and path.stem == stem and path.suffix.lower() in GEOTIFF_SUFFIXES
    ]
    assert matches, f"Missing GeoTIFF tile for stem '{stem}' in {tile_image_dir}"
    assert len(matches) == 1, f"Multiple GeoTIFF tiles found for stem '{stem}': {matches}"
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


def scale_box_edges(box: BoxRecord, pred_size: Tuple[int, int], image_size: Tuple[int, int]) -> Tuple[float, float, float, float]:
    pred_width, pred_height = pred_size
    image_width, image_height = image_size
    scale_x = image_width / pred_width
    scale_y = image_height / pred_height
    x_min = float(box["x_min"]) * scale_x
    y_min = float(box["y_min"]) * scale_y
    x_max = (float(box["x_max"]) + 1.0) * scale_x
    y_max = (float(box["y_max"]) + 1.0) * scale_y
    return x_min, y_min, x_max, y_max


def box_to_geojson_feature(
    box: BoxRecord,
    pred_shape: Tuple[int, int],
    tile_shape: Tuple[int, int],
    transform: GeoTransform,
    tile_path: Path,
) -> Dict[str, object]:
    pred_height, pred_width = pred_shape
    tile_height, tile_width = tile_shape
    x_min, y_min, x_max, y_max = scale_box_edges(box, (pred_width, pred_height), (tile_width, tile_height))

    top_left = transform.pixel_to_map(x_min, y_min)
    top_right = transform.pixel_to_map(x_max, y_min)
    bottom_right = transform.pixel_to_map(x_max, y_max)
    bottom_left = transform.pixel_to_map(x_min, y_max)
    ring = [top_left, top_right, bottom_right, bottom_left, top_left]

    properties = dict(box)
    properties["tile"] = tile_path.name
    properties["geo_x_min"] = min(point[0] for point in ring[:-1])
    properties["geo_y_min"] = min(point[1] for point in ring[:-1])
    properties["geo_x_max"] = max(point[0] for point in ring[:-1])
    properties["geo_y_max"] = max(point[1] for point in ring[:-1])

    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {
            "type": "Polygon",
            "coordinates": [[list(point) for point in ring]],
        },
    }


def contour_to_geojson_feature(
    contour: np.ndarray,
    component_id: int,
    pred_shape: Tuple[int, int],
    tile_shape: Tuple[int, int],
    transform: GeoTransform,
    tile_path: Path,
    pred_path: Path,
    prob: np.ndarray | None,
    min_area: int,
) -> GeoJsonFeature | None:
    pred_height, pred_width = pred_shape
    tile_height, tile_width = tile_shape
    scale_x = tile_width / pred_width
    scale_y = tile_height / pred_height

    component_mask = np.zeros(pred_shape, dtype=np.uint8)
    cv2.drawContours(component_mask, [contour], contourIdx=-1, color=1, thickness=-1)
    area = int(component_mask.sum())
    if area < min_area:
        return None

    points = contour.reshape(-1, 2).astype(np.float64)
    if len(points) < 3:
        return None

    ring: List[Tuple[float, float]] = []
    for col, row in points:
        tile_col = col * scale_x
        tile_row = row * scale_y
        ring.append(transform.pixel_to_map(tile_col, tile_row))

    if ring[0] != ring[-1]:
        ring.append(ring[0])
    if len(ring) < 4:
        return None

    ys, xs = np.nonzero(component_mask)
    properties: BoxRecord = {
        "image": pred_path.name,
        "stem": pred_path.stem,
        "tile": tile_path.name,
        "component_id": component_id,
        "area": area,
        "point_count": len(ring) - 1,
        "source": "contour",
    }
    if prob is not None and len(xs) > 0:
        component_probs = prob[ys, xs].astype(np.float32) / 255.0
        properties["score"] = float(component_probs.mean())
        properties["max_score"] = float(component_probs.max())

    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {
            "type": "Polygon",
            "coordinates": [[list(point) for point in ring]],
        },
    }


def skimage_contour_to_geojson_feature(
    contour: np.ndarray,
    component_mask: np.ndarray,
    component_id: int,
    pred_shape: Tuple[int, int],
    tile_shape: Tuple[int, int],
    transform: GeoTransform,
    tile_path: Path,
    pred_path: Path,
    prob: np.ndarray | None,
    min_area: int,
) -> GeoJsonFeature | None:
    area = int(component_mask.sum())
    if area < min_area or len(contour) < 3:
        return None

    pred_height, pred_width = pred_shape
    tile_height, tile_width = tile_shape
    scale_x = tile_width / pred_width
    scale_y = tile_height / pred_height

    ring: List[Tuple[float, float]] = []
    for row, col in contour:
        tile_col = float(col) * scale_x
        tile_row = float(row) * scale_y
        ring.append(transform.pixel_to_map(tile_col, tile_row))

    if ring[0] != ring[-1]:
        ring.append(ring[0])
    if len(ring) < 4:
        return None

    ys, xs = np.nonzero(component_mask)
    properties: BoxRecord = {
        "image": pred_path.name,
        "stem": pred_path.stem,
        "tile": tile_path.name,
        "component_id": component_id,
        "area": area,
        "point_count": len(ring) - 1,
        "source": "mask_contour",
    }
    if prob is not None and len(xs) > 0:
        component_probs = prob[ys, xs].astype(np.float32) / 255.0
        properties["score"] = float(component_probs.mean())
        properties["max_score"] = float(component_probs.max())

    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {
            "type": "Polygon",
            "coordinates": [[list(point) for point in ring]],
        },
    }


def box_to_polygon_feature(
    box: BoxRecord,
    pred_shape: Tuple[int, int],
    tile_shape: Tuple[int, int],
    transform: GeoTransform,
    tile_path: Path,
) -> GeoJsonFeature:
    feature = box_to_geojson_feature(box, pred_shape, tile_shape, transform, tile_path)
    feature["properties"]["source"] = "bbox_fallback"
    feature["properties"]["point_count"] = 4
    return feature


def mask_to_polygon_features_skimage(
    binary: np.ndarray,
    prob: np.ndarray | None,
    pred_path: Path,
    tile_path: Path,
    tile_shape: Tuple[int, int],
    transform: GeoTransform,
    min_area: int,
) -> List[GeoJsonFeature]:
    assert sk_measure is not None
    labels = sk_measure.label(binary, connectivity=2)
    features: List[GeoJsonFeature] = []

    for component_id in range(1, int(labels.max()) + 1):
        component_mask = labels == component_id
        if int(component_mask.sum()) < min_area:
            continue

        padded = np.pad(component_mask.astype(np.float32), pad_width=1, mode="constant", constant_values=0)
        contours = sk_measure.find_contours(padded, level=0.5)
        if not contours:
            continue

        contour = max(contours, key=len) - 1.0
        feature = skimage_contour_to_geojson_feature(
            contour=contour,
            component_mask=component_mask,
            component_id=component_id,
            pred_shape=binary.shape,
            tile_shape=tile_shape,
            transform=transform,
            tile_path=tile_path,
            pred_path=pred_path,
            prob=prob,
            min_area=min_area,
        )
        if feature is not None:
            features.append(feature)

    return features


def mask_to_polygon_features(
    pred: np.ndarray,
    prob: np.ndarray | None,
    pred_path: Path,
    tile_path: Path,
    tile_shape: Tuple[int, int],
    transform: GeoTransform,
    threshold: int,
    min_area: int,
    simplify_epsilon: float,
) -> List[GeoJsonFeature]:
    pred_shape = pred.shape
    binary = pred > threshold

    if cv2 is None and sk_measure is not None:
        return mask_to_polygon_features_skimage(
            binary=binary,
            prob=prob,
            pred_path=pred_path,
            tile_path=tile_path,
            tile_shape=tile_shape,
            transform=transform,
            min_area=min_area,
        )

    if cv2 is None:
        boxes = mask_to_boxes(pred=pred, prob=prob, threshold=threshold, min_area=min_area)
        for box in boxes:
            box["image"] = pred_path.name
            box["stem"] = pred_path.stem
        return [box_to_polygon_feature(box, pred_shape, tile_shape, transform, tile_path) for box in boxes]

    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features: List[GeoJsonFeature] = []
    for component_id, contour in enumerate(contours, start=1):
        if simplify_epsilon > 0:
            contour = cv2.approxPolyDP(contour, epsilon=float(simplify_epsilon), closed=True)
        feature = contour_to_geojson_feature(
            contour=contour,
            component_id=component_id,
            pred_shape=pred_shape,
            tile_shape=tile_shape,
            transform=transform,
            tile_path=tile_path,
            pred_path=pred_path,
            prob=prob,
            min_area=min_area,
        )
        if feature is not None:
            features.append(feature)
    return features


def save_geojson(features: List[GeoJsonFeature], save_path: Path, crs: str | None = None) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    collection: Dict[str, object] = {
        "type": "FeatureCollection",
        "features": features,
    }
    if crs is not None:
        collection["crs"] = {"type": "name", "properties": {"name": crs}}
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(collection, f, ensure_ascii=False, indent=2)


def save_gpkg(features: List[GeoJsonFeature], save_path: Path, crs: str | None = None, layer: str = "change_bboxes") -> None:
    if gpd is None or Polygon is None:
        raise ImportError("Saving GPKG requires geopandas and shapely. Install them or use --save_geojson.")

    rows = []
    geometries = []
    for feature in features:
        properties = dict(feature["properties"])
        coordinates = feature["geometry"]["coordinates"][0]
        geometries.append(Polygon(coordinates))
        rows.append(properties)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    geodata = gpd.GeoDataFrame(rows, geometry=geometries, crs=crs)
    geodata.to_file(save_path, layer=layer, driver="GPKG")


def save_polygon_csv(features: List[GeoJsonFeature], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "stem",
        "tile",
        "component_id",
        "area",
        "point_count",
        "score",
        "max_score",
        "source",
    ]
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for feature in features:
            writer.writerow(feature["properties"])


def save_qgis_polygon_qml(save_path: Path, fill_alpha: int = 90) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fill_alpha = max(0, min(255, int(fill_alpha)))
    qml = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28" styleCategories="Symbology">
  <renderer-v2 type="singleSymbol" symbollevels="0" enableorderby="0" forceraster="0">
    <symbols>
      <symbol type="fill" name="0" alpha="1" clip_to_extent="1">
        <layer class="SimpleFill" enabled="1" pass="0" locked="0">
          <Option type="Map">
            <Option name="color" type="QString" value="255,0,0,{fill_alpha}"/>
            <Option name="outline_color" type="QString" value="255,255,0,255"/>
            <Option name="outline_width" type="QString" value="0.4"/>
            <Option name="outline_width_unit" type="QString" value="MM"/>
            <Option name="style" type="QString" value="solid"/>
          </Option>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
  <layerOpacity>1</layerOpacity>
</qgis>
"""
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(qml)


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
    parser.add_argument("--tile_image_dir", type=str, default=None, help="Directory containing georeferenced GeoTIFF tiles.")
    parser.add_argument("--save_geojson", type=str, default=None, help="Save georeferenced bounding boxes for QGIS.")
    parser.add_argument("--save_gpkg", type=str, default=None, help="Save georeferenced bounding boxes as GeoPackage if geopandas is installed.")
    parser.add_argument("--gpkg_layer", type=str, default="change_bboxes", help="Layer name used when saving GeoPackage.")
    parser.add_argument("--save_polygon_geojson", type=str, default=None, help="Save georeferenced mask polygons for QGIS.")
    parser.add_argument("--save_polygon_gpkg", type=str, default=None, help="Save georeferenced mask polygons as GeoPackage if geopandas is installed.")
    parser.add_argument("--polygon_gpkg_layer", type=str, default="change_polygons", help="Layer name used for polygon GeoPackage.")
    parser.add_argument("--save_polygon_csv", type=str, default=None, help="Save polygon attribute statistics as CSV.")
    parser.add_argument("--save_polygon_qml", type=str, default=None, help="Save QGIS style file for translucent polygon highlight.")
    parser.add_argument("--polygon_simplify_epsilon", type=float, default=1.0, help="Contour simplification epsilon in pred-map pixels.")
    parser.add_argument("--polygon_fill_alpha", type=int, default=90, help="QGIS polygon fill alpha in [0,255].")
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
    tile_image_dir = Path(args.tile_image_dir) if args.tile_image_dir else None
    save_geojson_path = Path(args.save_geojson) if args.save_geojson else None
    save_gpkg_path = Path(args.save_gpkg) if args.save_gpkg else None
    save_polygon_geojson_path = Path(args.save_polygon_geojson) if args.save_polygon_geojson else None
    save_polygon_gpkg_path = Path(args.save_polygon_gpkg) if args.save_polygon_gpkg else None
    save_polygon_csv_path = Path(args.save_polygon_csv) if args.save_polygon_csv else None
    save_polygon_qml_path = Path(args.save_polygon_qml) if args.save_polygon_qml else None
    assert pred_dir.exists(), f"Missing pred_dir: {pred_dir}"
    if prob_dir is not None:
        assert prob_dir.exists(), f"Missing prob_dir: {prob_dir}"
    if image_dir is not None:
        assert image_dir.exists(), f"Missing image_dir: {image_dir}"
    if save_vis_dir is not None:
        assert image_dir is not None, "--save_vis_dir requires --image_dir"
    geo_outputs = [
        save_geojson_path,
        save_gpkg_path,
        save_polygon_geojson_path,
        save_polygon_gpkg_path,
        save_polygon_csv_path,
    ]
    if any(path is not None for path in geo_outputs):
        assert tile_image_dir is not None, "Geo outputs require --tile_image_dir"
    if tile_image_dir is not None:
        assert tile_image_dir.exists(), f"Missing tile_image_dir: {tile_image_dir}"

    records: List[BoxRecord] = []
    geojson_features: List[GeoJsonFeature] = []
    polygon_features: List[GeoJsonFeature] = []
    geojson_crs: str | None = None
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

        if tile_image_dir is not None and any(path is not None for path in geo_outputs):
            tile_path = find_geotiff_by_stem(tile_image_dir, pred_path.stem)
            transform, epsg, tile_shape = read_geotiff_metadata(tile_path)
            if geojson_crs is None and epsg is not None:
                geojson_crs = epsg
            pred = read_grayscale(pred_path)
            pred_shape = pred.shape

            if save_geojson_path is not None or save_gpkg_path is not None:
                for box in boxes:
                    geojson_features.append(
                        box_to_geojson_feature(
                            box=box,
                            pred_shape=pred_shape,
                            tile_shape=tile_shape,
                            transform=transform,
                            tile_path=tile_path,
                        )
                    )

            if save_polygon_geojson_path is not None or save_polygon_gpkg_path is not None or save_polygon_csv_path is not None:
                prob = None
                if prob_dir is not None:
                    prob_path = prob_dir / pred_path.name
                    assert prob_path.exists(), f"Missing probability map for {pred_path.name}: {prob_path}"
                    prob = read_grayscale(prob_path)
                    assert prob.shape == pred.shape, f"prob/pred size mismatch: {prob_path} vs {pred_path}"
                polygon_features.extend(
                    mask_to_polygon_features(
                        pred=pred,
                        prob=prob,
                        pred_path=pred_path,
                        tile_path=tile_path,
                        tile_shape=tile_shape,
                        transform=transform,
                        threshold=int(args.threshold),
                        min_area=int(args.min_area),
                        simplify_epsilon=float(args.polygon_simplify_epsilon),
                    )
                )

    save_csv(records, Path(args.save_csv))
    save_json(records, Path(args.save_json))
    if save_geojson_path is not None:
        save_geojson(geojson_features, save_geojson_path, geojson_crs)
    if save_gpkg_path is not None:
        save_gpkg(geojson_features, save_gpkg_path, geojson_crs, args.gpkg_layer)
    if save_polygon_geojson_path is not None:
        save_geojson(polygon_features, save_polygon_geojson_path, geojson_crs)
    if save_polygon_gpkg_path is not None:
        save_gpkg(polygon_features, save_polygon_gpkg_path, geojson_crs, args.polygon_gpkg_layer)
    if save_polygon_csv_path is not None:
        save_polygon_csv(polygon_features, save_polygon_csv_path)
    if save_polygon_qml_path is not None:
        save_qgis_polygon_qml(save_polygon_qml_path, args.polygon_fill_alpha)
    print(f"Saved {len(records)} boxes to {args.save_csv} and {args.save_json}")
    if save_geojson_path is not None:
        print(f"Saved {len(geojson_features)} georeferenced boxes to {save_geojson_path}")
    if save_gpkg_path is not None:
        print(f"Saved {len(geojson_features)} georeferenced boxes to {save_gpkg_path}")
    if save_polygon_geojson_path is not None:
        print(f"Saved {len(polygon_features)} georeferenced polygons to {save_polygon_geojson_path}")
    if save_polygon_gpkg_path is not None:
        print(f"Saved {len(polygon_features)} georeferenced polygons to {save_polygon_gpkg_path}")
    if save_polygon_qml_path is not None:
        print(f"Saved QGIS translucent polygon style to {save_polygon_qml_path}")


if __name__ == "__main__":
    main()
