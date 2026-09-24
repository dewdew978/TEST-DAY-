"""Inference for the Project 1 test day (Thai Character & Number Recognition, 72 classes).

VERSION "noembed" (ไม่ฝัง = not embedded): REQUIRES Project_1-data_dict.csv next to this script (or --data-dict PATH).

Reads every image in --test-dir (ts_img_001.jpg ...), predicts with a ResNet-18 checkpoint that
stores 'model_state' + 'class_to_idx', converts the predicted FOLDER number (e.g. 161) into the
3-digit class_id the grader expects (e.g. 101) using Project_1-data_dict.csv, and writes:

  * <out>            CSV  : gt_path, <group>   (rows in ts_img order, same as the submission sheet)
  * <out>.values.txt TXT  : the class_id column only, one per line  -> paste into the sheet

Usage:
    python predict_test_noembed.py --model ../CNN_font_and_nofont/model.pt --test-dir test_dataset --group NPC
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

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
NUM_CLASSES = 72
FALLBACK_FOLDER = "210"  # most frequent class; used only if an image cannot be opened (blank scores 0 anyway)


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


def natural_key(name):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def build_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
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

    folder_to_cid = load_data_dict(args.data_dict)
    if len(folder_to_cid) != NUM_CLASSES:
        sys.exit(f"data dict has {len(folder_to_cid)} classes, expected {NUM_CLASSES}")

    model, idx_to_folder = build_model(args.model, device)
    missing = set(idx_to_folder.values()) - set(folder_to_cid)
    if missing:
        sys.exit(f"checkpoint classes not in data dict: {sorted(missing)}")

    files = sorted((f for f in os.listdir(args.test_dir) if f.lower().endswith(IMG_EXT)), key=natural_key)
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
