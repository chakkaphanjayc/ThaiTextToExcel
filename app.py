from __future__ import annotations

from io import BytesIO
from typing import List

import pandas as pd
import streamlit as st

from ocr_utils import extract_text, items_to_dataframe


@st.cache_resource
def load_reader():
    import easyocr

    return easyocr.Reader(["th", "en"], gpu=False)


def merge_dataframes(existing: pd.DataFrame, new_frames: List[pd.DataFrame]) -> pd.DataFrame:
    frames = [existing] if existing is not None and not existing.empty else []
    frames.extend(df for df in new_frames if not df.empty)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def parse_existing_excel(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()
    return pd.read_excel(uploaded_file)


def dataframe_to_excel(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Extracted")
    return output.getvalue()


def render_item_table(items) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "text": item.text,
                "confidence": round(item.confidence, 3),
                "x_center": float(item.bbox[:, 0].mean()),
                "y_center": float(item.bbox[:, 1].mean()),
            }
            for item in items
        ]
    )


def main() -> None:
    st.set_page_config(page_title="Thai OCR to Excel", layout="wide")
    st.title("แปลงรูปภาษาไทยเป็นตาราง Excel")

    st.markdown(
        """
        - รองรับการเพิ่มรูปหลายใบพร้อมกัน
        - ตรวจจับว่าเป็นตารางหรือไม่ หากไม่ใช่จะเรียงตามตำแหน่งแกน x/y
        - รวมผลลัพธ์เข้ากับไฟล์ Excel เดิมได้
        """
    )

    reader = load_reader()

    col_left, col_right = st.columns([1, 1])
    with col_left:
        images = st.file_uploader(
            "อัปโหลดรูป (หลายไฟล์ได้)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )
        existing_excel = st.file_uploader(
            "เลือกไฟล์ Excel เดิม (ถ้าต้องการต่อข้อมูล)",
            type=["xlsx"],
        )

    with col_right:
        table_mode = st.selectbox(
            "โหมดตาราง",
            ["ตรวจจับอัตโนมัติ", "บังคับเป็นตาราง", "บังคับเรียงตามแกน x/y"],
        )
        confidence_threshold = st.slider(
            "ขั้นต่ำความมั่นใจ (confidence)",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
        )
        run_button = st.button("เริ่มแปลงข้อมูล")

    if run_button:
        if not images:
            st.warning("กรุณาอัปโหลดรูปก่อน")
            return

        existing_df = parse_existing_excel(existing_excel)
        extracted_frames: List[pd.DataFrame] = []
        all_items = []

        for image in images:
            result = extract_text(reader, image.getvalue())
            if table_mode == "บังคับเป็นตาราง":
                table_like = True
            elif table_mode == "บังคับเรียงตามแกน x/y":
                table_like = False
            else:
                table_like = result.table_like

            filtered_items = [
                item for item in result.items if item.confidence >= confidence_threshold
            ]
            df = items_to_dataframe(filtered_items, table_like)
            df.insert(0, "source_image", image.name)
            extracted_frames.append(df)
            all_items.extend(filtered_items)

        combined = merge_dataframes(existing_df, extracted_frames)

        st.subheader("ตัวอย่างข้อความที่อ่านได้")
        if all_items:
            st.dataframe(render_item_table(all_items), use_container_width=True)
        else:
            st.info("ไม่พบข้อความจากรูปที่อัปโหลด")

        st.subheader("ตารางข้อมูลที่ได้")
        if not combined.empty:
            st.dataframe(combined, use_container_width=True)
            excel_bytes = dataframe_to_excel(combined)
            st.download_button(
                "ดาวน์โหลดไฟล์ Excel",
                data=excel_bytes,
                file_name="thai_ocr_output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.info("ยังไม่มีข้อมูลสำหรับสร้าง Excel")

        st.subheader("ตัวอย่างภาพที่อัปโหลด")
        for image in images:
            st.image(image, caption=image.name, use_column_width=True)


if __name__ == "__main__":
    main()
