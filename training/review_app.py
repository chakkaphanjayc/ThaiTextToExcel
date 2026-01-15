from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

import easyocr
import streamlit as st
from PIL import Image


@st.cache_resource
def load_reader() -> easyocr.Reader:
    return easyocr.Reader(["th", "en"], gpu=False)


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


def save_labels(path: Path, rows: List[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "text"])
        for filename, text in rows:
            writer.writerow([filename, text])


st.set_page_config(page_title="OCR Review", layout="wide")

st.title("OCR Review & Correction")

with st.sidebar:
    labels_path = Path(
        st.text_input("Labels CSV", value="training/data/raw/labels.csv")
    )
    images_dir = Path(
        st.text_input("Images Dir", value="training/data/raw/images")
    )
    output_path = Path(
        st.text_input("Output CSV", value="training/data/raw/labels_reviewed.csv")
    )
    start_index = st.number_input("Start index", min_value=0, value=0, step=1)

if "rows" not in st.session_state:
    st.session_state.rows = []
if "labels" not in st.session_state or st.session_state.get("labels_path") != labels_path:
    st.session_state.labels = load_labels(labels_path)
    st.session_state.index = int(start_index)
    st.session_state.labels_path = labels_path

labels: List[Tuple[str, str]] = st.session_state.labels
index = st.session_state.index

if not labels:
    st.warning("No labels found. Please check the CSV path.")
    st.stop()

if index >= len(labels):
    st.success("Review complete!")
    if st.button("Save reviewed labels"):
        save_labels(output_path, st.session_state.rows)
        st.success(f"Saved to {output_path}")
    st.stop()

filename, original_text = labels[index]
image_path = images_dir / filename

if not image_path.exists():
    st.error(f"Image not found: {image_path}")
    st.stop()

image = Image.open(image_path)
reader = load_reader()
ocr_results = reader.readtext(str(image_path))
ocr_text = " ".join([text for _, text, _ in ocr_results]).strip()

col1, col2 = st.columns([1, 1])

with col1:
    st.image(image, caption=filename, use_column_width=True)

with col2:
    st.markdown("**Original label**")
    st.code(original_text or "(empty)")
    st.markdown("**EasyOCR prediction**")
    st.code(ocr_text or "(empty)")
    corrected = st.text_area("Corrected text", value=original_text, height=120)

save_col, skip_col, save_all_col = st.columns(3)

with save_col:
    if st.button("Save & Next"):
        st.session_state.rows.append((filename, corrected.strip()))
        st.session_state.index += 1
        st.experimental_rerun()

with skip_col:
    if st.button("Skip"):
        st.session_state.index += 1
        st.experimental_rerun()

with save_all_col:
    if st.button("Save progress"):
        save_labels(output_path, st.session_state.rows)
        st.success(f"Saved to {output_path}")
