from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path
from typing import List, Tuple


def load_labels(path: Path) -> List[Tuple[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            filename = (row.get("filename") or "").strip()
            text = (row.get("text") or "").strip()
            if filename:
                rows.append((filename, text))
        return rows


def write_labels(path: Path, rows: List[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "text"])
        for filename, text in rows:
            writer.writerow([filename, text])


def copy_images(rows: List[Tuple[str, str]], source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename, _ in rows:
        src = source_dir / filename
        dst = target_dir / filename
        if not dst.exists():
            shutil.copy2(src, dst)


def split_rows(rows: List[Tuple[str, str]], val_ratio: float) -> tuple[list, list]:
    shuffled = rows[:]
    random.shuffle(shuffled)
    split_index = int(len(shuffled) * (1 - val_ratio))
    return shuffled[:split_index], shuffled[split_index:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare OCR dataset")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    rows = load_labels(args.labels)
    train_rows, val_rows = split_rows(rows, args.val_ratio)

    train_dir = args.output_dir / "train"
    val_dir = args.output_dir / "val"

    write_labels(train_dir / "labels.csv", train_rows)
    write_labels(val_dir / "labels.csv", val_rows)

    copy_images(train_rows, args.images_dir, train_dir / "images")
    copy_images(val_rows, args.images_dir, val_dir / "images")

    print(f"Prepared {len(train_rows)} training rows and {len(val_rows)} validation rows")


if __name__ == "__main__":
    main()
