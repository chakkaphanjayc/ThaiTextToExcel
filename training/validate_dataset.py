from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ValidationReport:
    total_rows: int
    missing_images: int
    empty_text_rows: int
    duplicate_filenames: int
    unique_characters: int
    most_common_characters: list[tuple[str, int]]
    examples_missing_images: list[str]
    examples_empty_text: list[str]


def read_labels(labels_path: Path) -> list[tuple[str, str]]:
    with labels_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            filename = (row.get("filename") or "").strip()
            text = (row.get("text") or "").strip()
            if filename:
                rows.append((filename, text))
        return rows


def validate(labels_path: Path, images_dir: Path, report_path: Path) -> ValidationReport:
    rows = read_labels(labels_path)
    missing_images = []
    empty_text = []
    char_counter: Counter[str] = Counter()
    filename_counter: Counter[str] = Counter()

    for filename, text in rows:
        filename_counter[filename] += 1
        if not (images_dir / filename).exists():
            missing_images.append(filename)
        if not text:
            empty_text.append(filename)
        char_counter.update(list(text))

    duplicates = sum(1 for _, count in filename_counter.items() if count > 1)
    report = ValidationReport(
        total_rows=len(rows),
        missing_images=len(missing_images),
        empty_text_rows=len(empty_text),
        duplicate_filenames=duplicates,
        unique_characters=len(char_counter),
        most_common_characters=char_counter.most_common(20),
        examples_missing_images=missing_images[:20],
        examples_empty_text=empty_text[:20],
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(report), handle, ensure_ascii=False, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate OCR labels and images")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    report = validate(args.labels, args.images_dir, args.report)
    print("Validation complete:")
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
