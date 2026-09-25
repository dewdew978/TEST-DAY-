"""Inference for the Project 1 test day (Thai Character & Number Recognition, 72 classes).

VERSION "embed" (ฝัง = embedded): has a built-in copy of the class_id table -> runs even without Project_1-data_dict.csv.

Reads every image in --test-dir (ts_img_001.jpg ...), predicts with a ResNet-18 checkpoint that
stores 'model_state' + 'class_to_idx', converts the predicted FOLDER number (e.g. 161) into the
3-digit class_id the grader expects (e.g. 101) using Project_1-data_dict.csv, and writes:

  * <out>            CSV  : gt_path, <group>   (rows in ts_img order, same as the submission sheet)
  * <out>.values.txt TXT  : the class_id column only, one per line  -> paste into the sheet

Usage:
    python predict_test_embed.py --model ../CNN_font_and_nofont/model.pt --test-dir test_dataset --group NPC
"""
import argparse
import csv
import io
import os
import re
import sys
import time

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

for _stream in (sys.stdout, sys.stderr):   # a Thai/odd path in a message must not crash the run
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif")
NUM_CLASSES = 72
FALLBACK_FOLDER = "210"  # most frequent class; used only if an image cannot be opened (blank scores 0 anyway)

# Built-in copy of Project_1-data_dict.csv (folder -> class_id), generated from that file.
# Used ONLY if the CSV is missing or unusable. If the CSV is present it always wins.
BUILTIN_TABLE = {
    "161": "101",  # ก
    "162": "102",  # ข
    "163": "103",  # ฃ
    "164": "104",  # ค
    "167": "105",  # ง
    "168": "106",  # จ
    "169": "107",  # ฉ
    "170": "108",  # ช
    "171": "109",  # ซ
    "173": "110",  # ญ
    "175": "111",  # ฏ
    "176": "112",  # ฐ
    "177": "113",  # ฑ
    "178": "114",  # ฒ
    "179": "115",  # ณ
    "180": "116",  # ด
    "181": "117",  # ต
    "182": "118",  # ถ
    "183": "119",  # ท
    "184": "120",  # ธ
    "185": "121",  # น
    "186": "122",  # บ
    "187": "123",  # ป
    "188": "124",  # ผ
    "189": "125",  # ฝ
    "190": "126",  # พ
    "191": "127",  # ฟ
    "192": "128",  # ภ
    "193": "129",  # ม
    "194": "130",  # ย
    "195": "131",  # ร
    "196": "132",  # ฤ
    "197": "133",  # ล
    "199": "134",  # ว
    "200": "135",  # ศ
    "201": "136",  # ษ
    "202": "137",  # ส
    "203": "138",  # ห
    "204": "139",  # ฬ
    "205": "140",  # อ
    "206": "141",  # ฮ
    "207": "201",  # ฯ
    "209": "202",  # ั
    "210": "203",  # า
    "212": "204",  # ิ
    "213": "205",  # ี
    "214": "206",  # ึ
    "215": "207",  # ื
    "216": "208",  # ุ
    "217": "209",  # ู
    "224": "210",  # เ
    "225": "211",  # แ
    "226": "212",  # โ
    "227": "213",  # ใ
    "228": "214",  # ไ
    "229": "215",  # ๅ
    "230": "216",  # ๆ
    "231": "217",  # ็
    "232": "218",  # ่
    "233": "219",  # ้
    "234": "220",  # ๊
    "236": "221",  # ์
    "240": "301",  # ๐
    "241": "302",  # ๑
    "242": "303",  # ๒
    "243": "304",  # ๓
    "244": "305",  # ๔
    "245": "306",  # ๕
    "246": "307",  # ๖
    "247": "308",  # ๗
    "248": "309",  # ๘
    "249": "310",  # ๙
}


def load_data_dict(path):
    """folder -> class_id. The CSV is cp874 (TIS-620) encoded, so decode explicitly."""
    raw = open(path, "rb").read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp874")
    mapping = {}
    for row in csv.reader(io.StringIO(text)):
        if len(row) >= 3 and row[0].strip().isdigit() and row[2].strip().isdigit():
            mapping[row[0].strip()] = row[2].strip()
    return mapping


def resolve_table(path):
    """CSV first (so a corrected table from the teacher just works); built-in copy only as a fallback."""
    table = None
    if os.path.exists(path):
        try:
            table = load_data_dict(path)
        except Exception as e:
            print(f"WARNING cannot read {path} ({e!r}) -> using BUILT-IN table")
        else:
            if len(table) != NUM_CLASSES:
                print(f"WARNING {path} has {len(table)} classes, expected {NUM_CLASSES} -> using BUILT-IN table")
                table = None
    else:
        print(f"WARNING {path} not found -> using BUILT-IN table")
    if table is None:
        print("class_id table: BUILT-IN (embedded in this script)")
        return dict(BUILTIN_TABLE)
    diff = sorted(f for f in set(table) | set(BUILTIN_TABLE) if table.get(f) != BUILTIN_TABLE.get(f))
    if diff:
        print(f"class_id table: CSV file ({path}); NOTE it differs from the built-in copy in {len(diff)} folder(s): {diff[:10]}")
    else:
        print("class_id table: CSV file (identical to the built-in copy)")
    return table


def natural_key(name):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def build_model(ckpt_path, device):
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:   # torch < 1.13 does not know weights_only
        ckpt = torch.load(ckpt_path, map_location=device)
    if not (isinstance(ckpt, dict) and "model_state" in ckpt and "class_to_idx" in ckpt):
        sys.exit(f"{ckpt_path}: expected a checkpoint with 'model_state' and 'class_to_idx'")
    model = models.resnet18()
    model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(model.fc.in_features, NUM_CLASSES))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    idx_to_folder = {int(idx): str(folder) for folder, idx in ckpt["class_to_idx"].items()}
    return model, idx_to_folder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="checkpoint .pt (model_state + class_to_idx)")
    ap.add_argument("--test-dir", default="test_dataset")
    ap.add_argument("--data-dict", default=os.path.join(HERE, "Project_1-data_dict.csv"))
    ap.add_argument("--group", default="NPC", help="column header for the group (our sheet column)")
    ap.add_argument("--out", default="predictions_NPC.csv")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folder_to_cid = resolve_table(args.data_dict)

    model, idx_to_folder = build_model(args.model, device)
    missing = set(idx_to_folder.values()) - set(folder_to_cid)
    if missing:
        sys.exit(f"checkpoint classes not in data dict: {sorted(missing)}")

    if not os.path.isdir(args.test_dir):
        sys.exit(f"test folder not found: {args.test_dir}  (run from the right folder or pass --test-dir PATH)")
    all_files = sorted((f for f in os.listdir(args.test_dir) if os.path.isfile(os.path.join(args.test_dir, f))), key=natural_key)
    files = [f for f in all_files if f.lower().endswith(IMG_EXT)]
    if not files:   # e.g. images saved without an extension: try every visible file, unreadable ones are guessed later
        files = [f for f in all_files if "." not in f]
        if files:
            print(f"WARNING no files with an image extension in {args.test_dir}; using {len(files)} file(s) without extension")
    ignored = [f for f in all_files if f not in set(files)]
    if ignored:
        print(f"NOTE ignored {len(ignored)} non-image file(s): {ignored[:5]}")
    if not files:
        sys.exit(f"no images found in {args.test_dir}")

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    preds, failed = [], []
    with torch.no_grad():
        for i in range(0, len(files), args.batch_size):
            names = files[i:i + args.batch_size]
            tensors, ok = [], []
            for n in names:
                try:
                    tensors.append(tf(Image.open(os.path.join(args.test_dir, n)).convert("RGB")))
                    ok.append(True)
                except Exception as e:  # unreadable image: keep the row, guess a class
                    failed.append((n, repr(e)))
                    ok.append(False)
            out_idx = iter(model(torch.stack(tensors).to(device)).argmax(1).tolist()) if tensors else iter(())
            for good in ok:
                folder = idx_to_folder[next(out_idx)] if good else FALLBACK_FOLDER
                preds.append(folder_to_cid[folder])

    valid = set(folder_to_cid.values())
    assert len(preds) == len(files) and all(p in valid and len(p) == 3 for p in preds)

    stem = lambda n: "./test_dataset/" + os.path.splitext(n)[0]
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gt_path", args.group])
        for n, p in zip(files, preds):
            w.writerow([stem(n), p])
    with open(args.out + ".values.txt", "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(preds) + "\n")

    print(f"device={device}  images={len(files)}  time={time.time() - t0:.1f}s")
    print(f"first: {files[0]} -> {preds[0]}   last: {files[-1]} -> {preds[-1]}")
    print(f"wrote {args.out} and {args.out}.values.txt")
    if failed:
        print(f"WARNING {len(failed)} unreadable image(s), guessed {folder_to_cid[FALLBACK_FOLDER]}:", failed[:5])
    if not 450 <= len(files) <= 550:
        print(f"WARNING expected about 500 images, found {len(files)}")


if __name__ == "__main__":
    main()
