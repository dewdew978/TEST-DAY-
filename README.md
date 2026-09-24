# Project1_TestDay_Kit — ชุดเตรียมวันทดสอบ Project 1 (Thai Character & Number Recognition)

แตกไฟล์แล้วปล่อยให้ 4 อย่างนี้อยู่ **โฟลเดอร์เดียวกัน (วางเป็นพี่น้องกัน อย่าซ้อนกัน อย่าเปลี่ยนชื่อ `rehearsal`)**

```
Project1_TestDay_Kit/
├─ CNN2_test_day/      โมเดลของ CNN2 (TAR) + สคริปต์วันแข่ง
├─ FOURTH_test_day/    โมเดลของ FOURTH + สคริปต์วันแข่ง
├─ rehearsal/          ชุดซ้อม 500 ภาพ ครบ 72 คลาส พร้อมเฉลย + คู่มือ (rehearsal/README.md)
├─ rehearsal.py        ตัวตรวจคะแนนชุดซ้อม (ต้องอยู่ข้าง ๆ โฟลเดอร์ rehearsal)
└─ README.md           ไฟล์นี้
```

## ต้องมีอะไรในเครื่อง
Python + `torch`, `torchvision`, `pillow` — ลอง `python -c "import torch, torchvision, PIL"` ต้องไม่ขึ้น error
(ถ้า error: `pip install torch torchvision pillow`)

## ซ้อมก่อนวันแข่ง (ทำได้เลย ~1 นาที)
เปิด terminal ที่โฟลเดอร์โมเดลที่อยากลอง แล้วรัน 2 คำสั่ง:
```
cd CNN2_test_day                (หรือ FOURTH_test_day)
python predict_test_embed.py --model model.pt --test-dir ../rehearsal/test_dataset --group NPC --out pred.csv
python ../rehearsal.py score pred.csv.values.txt
```
จะเห็น `overall: …/500` และรายการภาพที่ตอบผิด · แถว `trained_by=none` คือ **ภาพที่ทั้งสองโมเดลไม่เคยฝึก** (วัดผลได้จริงที่สุด)
ผลที่เครื่องต้นทางได้: **CNN2 489/500 (97.80%)**, **FOURTH 494/500 (98.80%)** ถ้าเครื่องคุณได้ต่างมาก ให้แจ้ง

## วันแข่งจริง
ใช้เฉพาะโฟลเดอร์ `CNN2_test_day` หรือ `FOURTH_test_day` (ตามโมเดลที่ทีมเลือกส่ง) — อ่าน `README.md` ในโฟลเดอร์นั้น:
วางโฟลเดอร์ `test_dataset` ที่ได้รับวันแข่ง → ดับเบิลคลิก `run_embed.bat` → เอาไฟล์ `predictions_NPC.csv.values.txt` ไปวางคอลัมน์ **NPC** ในชีต
- **ห้ามก๊อปภาพจาก `rehearsal/` ไปวางเป็น `test_dataset` ของชุดจริง** ชุดซ้อมมีไว้ซ้อมเท่านั้น
- ตารางส่งผลต้องกรอก **class_id 3 หลัก** ไม่ใช่เลขโฟลเดอร์ (สคริปต์แปลงให้แล้ว)

## ตรวจว่าได้ไฟล์ครบและไม่เสีย
| ไฟล์ | MD5 ที่ต้องได้ (`certutil -hashfile <ไฟล์> MD5` ใน Windows / `md5sum <ไฟล์>` ใน Linux-Mac) |
|---|---|
| `CNN2_test_day/model.pt` | `41e085d32cd8fb68a75b76e16492354c` |
| `FOURTH_test_day/model.pt` | `b29f8c3c725c8f334e1f7149193dae56` |

`rehearsal/test_dataset/` ต้องมี 500 ภาพ และ `rehearsal/answer_key.csv` ต้องมี 500 แถว (ไม่นับหัวตาราง)

## ที่ไม่ได้รวมมา
ชุดข้อมูลเต็ม (`dataset/round2` 62,707 ภาพ) — จึงรัน `rehearsal.py build`, `eval`, `verify` ไม่ได้ ถ้าต้องการสร้างชุดซ้อมใหม่หรือวัดรายคลาส
ให้ดูวิธีใน `rehearsal/README.md` (หัวข้อวิธี C) ส่วนการซ้อมและตรวจคะแนนด้านบนใช้ได้ครบโดยไม่ต้องมีชุดข้อมูลเต็ม
