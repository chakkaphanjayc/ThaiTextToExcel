from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Iterable, List

import cv2
import numpy as np
import pandas as pd
from PIL import Image


@dataclass
class OcrItem:
    text: str
    confidence: float
    bbox: np.ndarray


@dataclass
class OcrResult:
    items: List[OcrItem]
    table_like: bool


def load_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def detect_table_like(image: np.ndarray) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=120, minLineLength=80, maxLineGap=8)

    if lines is None:
        return False

    vertical = 0
    horizontal = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x1 - x2) < 8:
            vertical += 1
        if abs(y1 - y2) < 8:
            horizontal += 1

    return vertical >= 3 and horizontal >= 3


def run_easyocr(reader, image: np.ndarray) -> List[OcrItem]:
    results = reader.readtext(image)
    items: List[OcrItem] = []
    for bbox, text, confidence in results:
        items.append(OcrItem(text=text.strip(), confidence=float(confidence), bbox=np.array(bbox)))
    return [item for item in items if item.text]


def cluster_positions(values: Iterable[float], gap: float) -> List[float]:
    sorted_values = sorted(values)
    if not sorted_values:
        return []

    clusters = [[sorted_values[0]]]
    for value in sorted_values[1:]:
        if abs(value - clusters[-1][-1]) <= gap:
            clusters[-1].append(value)
        else:
            clusters.append([value])

    return [float(np.mean(cluster)) for cluster in clusters]


def assign_cluster(value: float, clusters: List[float]) -> int:
    if not clusters:
        return 0
    distances = [abs(value - center) for center in clusters]
    return int(np.argmin(distances))


def items_to_dataframe(items: List[OcrItem], table_like: bool) -> pd.DataFrame:
    if not items:
        return pd.DataFrame()

    centers = np.array(
        [
            (float(np.mean(item.bbox[:, 0])), float(np.mean(item.bbox[:, 1])))
            for item in items
        ]
    )
    xs = centers[:, 0]
    ys = centers[:, 1]

    y_gap = max(12.0, float(np.percentile(np.diff(np.sort(ys)), 60))) if len(ys) > 1 else 15.0
    row_centers = cluster_positions(ys, gap=y_gap)

    if table_like:
        x_gap = max(20.0, float(np.percentile(np.diff(np.sort(xs)), 60))) if len(xs) > 1 else 40.0
        col_centers = cluster_positions(xs, gap=x_gap)
    else:
        col_centers = []

    rows = []
    for item, (cx, cy) in zip(items, centers):
        row_idx = assign_cluster(cy, row_centers)
        col_idx = assign_cluster(cx, col_centers)
        rows.append((row_idx, col_idx, item.text, cx, cy))

    rows.sort(key=lambda row: (row[0], row[1], row[3]))
    if not table_like:
        return pd.DataFrame(
            [
                {"text": text, "x_center": cx, "y_center": cy}
                for _, _, text, cx, cy in rows
            ]
        )

    max_row = max(row[0] for row in rows) + 1
    max_col = max(row[1] for row in rows) + 1

    table = [["" for _ in range(max_col)] for _ in range(max_row)]
    for row_idx, col_idx, text, _, _ in rows:
        existing = table[row_idx][col_idx]
        table[row_idx][col_idx] = f"{existing} {text}".strip() if existing else text

    df = pd.DataFrame(table)

    return df


def extract_text(reader, image_bytes: bytes) -> OcrResult:
    image = load_image(image_bytes)
    table_like = detect_table_like(image)
    items = run_easyocr(reader, image)
    return OcrResult(items=items, table_like=table_like)
