# OCR Training Pipeline (Handwritten Thai)

เอกสารนี้อธิบาย pipeline สำหรับเตรียมข้อมูล, ตรวจสอบความถูกต้อง, และ train โมเดล OCR แบบปรับแต่งเอง
เพื่อให้รองรับลายมือภาษาไทยได้ดีขึ้น โดยใช้โครงสร้างข้อมูลภายในโฟลเดอร์ `training/`.

> หมายเหตุ: Pipeline นี้เน้นการฝึกโมเดล recognition (อ่านข้อความจากภาพ) แบบ line-level
> หากมีข้อมูลที่เป็นภาพทั้งเอกสาร ให้ตัด (crop) เป็นบรรทัดหรือคำก่อน

## โครงสร้างโฟลเดอร์

```
training/
  data/
    raw/
      images/
        0001.png
        0002.png
      labels.csv
    processed/
      train/
        images/
        labels.csv
      val/
        images/
        labels.csv
    reports/
  models/
  review_app.py
  validate_dataset.py
  prepare_dataset.py
  train_recognizer.py
  evaluate_recognizer.py
```

## 1) เตรียมข้อมูล

1. นำภาพทั้งหมดมาไว้ที่ `training/data/raw/images/`
2. สร้างไฟล์ `training/data/raw/labels.csv` ในรูปแบบ:

```
filename,text
0001.png,สวัสดีครับ
0002.png,เลขที่ 1234
```

## 2) ตรวจสอบข้อมูลเบื้องต้น (Validation)

รันคำสั่ง:

```bash
python training/validate_dataset.py \
  --labels training/data/raw/labels.csv \
  --images-dir training/data/raw/images \
  --report training/data/reports/validation_report.json
```

ผลลัพธ์จะสรุปจำนวนไฟล์ที่ขาดหาย, บรรทัดที่ข้อความว่าง, อักขระที่พบบ่อย ฯลฯ

## 3) ระบบตรวจสอบความถูกต้อง (Review App)

รัน Streamlit app เพื่อช่วยตรวจสอบ/แก้ไข label:

```bash
streamlit run training/review_app.py
```

ฟีเจอร์หลัก:
- แสดงภาพ + OCR ที่คาดการณ์จาก EasyOCR
- สามารถแก้ไขข้อความ แล้วบันทึกเป็นไฟล์ CSV ใหม่
- ช่วยตรวจสอบความถูกต้องก่อนนำไป train

เอาต์พุต: `training/data/raw/labels_reviewed.csv`

## 4) เตรียมชุดข้อมูล Train/Validation

```bash
python training/prepare_dataset.py \
  --labels training/data/raw/labels_reviewed.csv \
  --images-dir training/data/raw/images \
  --output-dir training/data/processed \
  --val-ratio 0.1
```

สคริปต์จะสุ่มแบ่ง train/val และคัดลอกไฟล์ไปยังโฟลเดอร์ย่อย พร้อมสร้าง `labels.csv`

## 5) Train โมเดล Recognition

```bash
python training/train_recognizer.py \
  --train-labels training/data/processed/train/labels.csv \
  --train-images training/data/processed/train/images \
  --val-labels training/data/processed/val/labels.csv \
  --val-images training/data/processed/val/images \
  --output-dir training/models \
  --epochs 20 \
  --batch-size 16
```

เอาต์พุต:
- `training/models/ocr_crnn.pt` (น้ำหนักโมเดล)
- `training/models/vocab.json` (ชุดอักขระที่ใช้)

## 6) ประเมินโมเดล

```bash
python training/evaluate_recognizer.py \
  --labels training/data/processed/val/labels.csv \
  --images-dir training/data/processed/val/images \
  --model-path training/models/ocr_crnn.pt \
  --vocab-path training/models/vocab.json
```

จะแสดง CER/WER เพื่อประเมินคุณภาพการอ่าน

## หมายเหตุด้านคุณภาพข้อมูล

- ภาพลายมือควรคมชัด ไม่มีแสงสะท้อน และมีคอนทราสต์สูง
- ควรมีความหลากหลายของลายมือและรูปแบบตัวอักษร
- ถ้า CER/WER สูง ให้เพิ่มข้อมูลหรือใช้ review app เพื่อแก้ไข label ให้แม่นยำขึ้น
