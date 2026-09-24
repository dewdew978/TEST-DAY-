"""Rehearsal tools for the Project 1 test day (workspace level, NOT part of any git repo).

  python rehearsal.py build     make a 500-image rehearsal set (all 72 classes, ts_img_XXX.jpg, shuffled) + answer key
  python rehearsal.py eval      per-class accuracy of CNN2 / FOURTH / FOURTH-nofont on the images NEITHER model trained on
  python rehearsal.py verify    prove the rebuilt 20% splits equal the ones used in training (exact-count check)
  python rehearsal.py score F   score predictions (predictions_NPC.csv.values.txt or predictions_NPC.csv) against the answer key

Nothing is hard-coded: put options BEFORE the command, e.g.
  python rehearsal.py --cnn2 <CNN2_DIR> --fourth <FOURTH_DIR> --out <OUTPUT_DIR> build
  python rehearsal.py --key <answer_key.csv> score <predictions file>
Defaults are worked out from where this script lives (its parent folder holds CNN2/ and CNN_font_and_nofont/).

"Unseen" = the 20% real-image validation split of each repo, rebuilt with the SAME code and seed as its TrainingCNN.py:
  CNN2   : torch random_split(ImageFolder(dataset/round2), [80%, 20%], Generator().manual_seed(42))
  FOURTH : stratified_split(ImageFolder(dataset/round2).samples, 0.2, 42)   (copied verbatim from its TrainingCNN.py)
A rehearsal image is "clean" if it is in BOTH validation splits (neither model trained on it). Rare classes cannot
be filled from clean images alone, so they are topped up with images a model HAS seen; each row records that in
`trained_by` so those rows are never mixed into the clean accuracy.
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
# defaults (relative to this script); every one can be overridden from the command line, see configure()
CNN2 = os.path.join(WS, "CNN2")
FOURTH = os.path.join(WS, "CNN_font_and_nofont")
OUT = os.path.join(HERE, "rehearsal")
KEY = None            # answer key path; None -> <OUT>/answer_key.csv
PREDICT_SCRIPT = os.path.join(HERE, "predict_test_embed.py")
TRAIN_SEED = 42       # seed used by both TrainingCNN.py files for the 80/20 split: NEVER change (rebuilds their exact val sets)
SEED = 42             # sampling seed for `build` only (--seed); does not affect which images are 'unseen'
N_TOTAL = 500
MODELS = {}           # name -> checkpoint, filled by configure()


def configure(a):
    """Apply command-line overrides to the module-level settings."""
    global CNN2, FOURTH, OUT, KEY, PREDICT_SCRIPT, SEED, N_TOTAL, MODELS
    CNN2 = os.path.abspath(a.cnn2) if a.cnn2 else CNN2
    FOURTH = os.path.abspath(a.fourth) if a.fourth else FOURTH
    OUT = os.path.abspath(a.out) if a.out else OUT
    KEY = os.path.abspath(a.key) if a.key else None
    PREDICT_SCRIPT = os.path.abspath(a.predict_script) if a.predict_script else PREDICT_SCRIPT
    SEED = a.seed
    N_TOTAL = a.n
    MODELS = {
        "cnn2": os.path.join(CNN2, "model.pt"),
        "fourth": os.path.join(FOURTH, "model.pt"),
        "fourth_nofont": os.path.join(FOURTH, "model_nofont.pt"),
    }
    for item in a.model or []:          # --model NAME=PATH replaces or adds a checkpoint
        name, _, path = item.partition("=")
        MODELS[name] = os.path.abspath(path)
    MODELS = {n: c for n, c in MODELS.items() if os.path.exists(c)}


TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ------------------------------------------------------------------ helpers
def load_pt():
    """Reuse the exact model builder / class table of the test-day script."""
    if not os.path.exists(PREDICT_SCRIPT):
        sys.exit(f"cannot find {PREDICT_SCRIPT} (use --predict-script PATH)")
    spec = importlib.util.spec_from_file_location("pt_embed", PREDICT_SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def key_of(path, root):
    return os.path.relpath(path, root).replace("\\", "/")


def label_of(folder):
    return bytes([int(folder)]).decode("cp874")


def imagefolder(repo):
    root = os.path.join(repo, "dataset", "round2")
    ds = ImageFolder(root)
    return root, ds


def cnn2_val_keys():
    root, ds = imagefolder(CNN2)
    n = len(ds)
    train_size = int(0.8 * n)
    generator = torch.Generator().manual_seed(TRAIN_SEED)
    _, val = random_split(ds, [train_size, n - train_size], generator=generator)
    return {key_of(ds.samples[i][0], root) for i in val.indices}, {key_of(p, root): ds.classes[y] for p, y in ds.samples}


def stratified_split(samples, val_frac, seed):  # verbatim from CNN_font_and_nofont/TrainingCNN.py
    """Split per class so rare classes appear in both train and val (when they have >=2 images)."""
    by_class = defaultdict(list)
    for s in samples:
        by_class[s[1]].append(s)
    rng = random.Random(seed)
    train, val = [], []
    for label in sorted(by_class):
        items = by_class[label]
        rng.shuffle(items)
        n_val = min(int(round(len(items) * val_frac)), len(items) - 1)
        val += items[:n_val]
        train += items[n_val:]
    return train, val


def fourth_val_keys():
    root, ds = imagefolder(FOURTH)
    _, val = stratified_split(ds.samples, 0.2, TRAIN_SEED)
    return {key_of(p, root) for p, _ in val}, {key_of(p, root): ds.classes[y] for p, y in ds.samples}


def splits():
    val_c, all_c = cnn2_val_keys()
    val_f, all_f = fourth_val_keys()
    assert all_c == all_f, "the two dataset/round2 folders differ (different files or classes)"
    return val_c, val_f, all_c   # all_c: key -> class folder name


def predict(model, idx_to_folder, paths, bs=128, tag=""):
    out, t0 = [], time.time()
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            batch = [TF(Image.open(p).convert("RGB")) for p in paths[i:i + bs]]
            out += [idx_to_folder[j] for j in model(torch.stack(batch)).argmax(1).tolist()]
            if tag and (i // bs) % 20 == 19:
                print(f"   {tag}: {len(out)}/{len(paths)}  ({time.time() - t0:.0f}s)", flush=True)
    return out


def load_models(pt, names):
    dev = torch.device("cpu")
    return {n: pt.build_model(MODELS[n], dev) for n in names}


def img_path(key):  # both repos hold identical files (checked in splits()); read from CNN2's copy
    return os.path.join(CNN2, "dataset", "round2", key)


# ------------------------------------------------------------------ build
def cmd_build(args):
    pt = load_pt()
    folder_to_cid = pt.BUILTIN_TABLE
    val_c, val_f, all_folder = splits()
    pools = defaultdict(list)  # folder -> keys, ordered tier1 (clean) -> tier2 -> tier3
    rng = random.Random(SEED)
    tiers = defaultdict(lambda: defaultdict(list))
    for key, folder in sorted(all_folder.items()):
        in_c, in_f = key in val_c, key in val_f
        tier = 1 if (in_c and in_f) else (2 if (in_c or in_f) else 3)
        tiers[folder][tier].append(key)
    trained_by = {}
    for key in all_folder:
        in_c, in_f = key in val_c, key in val_f
        trained_by[key] = "none" if (in_c and in_f) else ("fourth" if in_c else ("cnn2" if in_f else "both"))
        # in_c = CNN2 did NOT train on it -> only FOURTH did; in_f = FOURTH did NOT train -> only CNN2 did
    for folder in tiers:
        lst = []
        for t in (1, 2, 3):
            items = tiers[folder][t][:]
            rng.shuffle(items)
            lst += items
        pools[folder] = lst
    folders = sorted(pools)
    assert len(folders) == 72 and set(folders) == set(folder_to_cid)

    # round-robin: every class gets one image per pass until N_TOTAL (classes that run dry drop out)
    taken = {f: 0 for f in folders}
    total = 0
    while total < N_TOTAL:
        order = [f for f in folders if taken[f] < len(pools[f])]
        rng.shuffle(order)
        if not order:
            break
        for f in order:
            if total >= N_TOTAL:
                break
            taken[f] += 1
            total += 1
    chosen = [k for f in folders for k in pools[f][:taken[f]]]
    rng.shuffle(chosen)
    assert len(chosen) == N_TOTAL

    dst = os.path.join(OUT, "test_dataset")
    if os.path.isdir(dst):   # only ever replace a folder this tool made: files named ts_img_NNN.jpg and nothing else
        foreign = [f for f in os.listdir(dst) if not (f.startswith("ts_img_") and f.lower().endswith(".jpg"))]
        if foreign:
            sys.exit(f"refusing to overwrite {dst}: it contains files this tool did not create (e.g. {foreign[:3]}). Pick another --out.")
        shutil.rmtree(dst)
    os.makedirs(dst)
    rows = []
    for i, key in enumerate(chosen, 1):
        name = f"ts_img_{i:03d}.jpg"
        shutil.copyfile(img_path(key), os.path.join(dst, name))
        folder = all_folder[key]
        rows.append([f"./test_dataset/{name}", folder_to_cid[folder], folder, trained_by[key], key])
    with open(os.path.join(OUT, "answer_key.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gt_path", "gt_class_id", "folder", "trained_by", "source"])
        w.writerows(rows)

    per = Counter(r[2] for r in rows)
    tb = Counter(r[3] for r in rows)
    clean = tb["none"]
    print(f"rehearsal set: {len(rows)} images, {len(per)} classes, per-class min/max = {min(per.values())}/{max(per.values())}")
    print(f"clean (neither model trained on it): {clean}   trained_by counts: {dict(tb)}")
    short = sorted(((folder_to_cid[f], label_of(f), taken[f], sum(1 for k in pools[f][:taken[f]] if trained_by[k] != 'none')) for f in folders if taken[f] < 6 or any(trained_by[k] != 'none' for k in pools[f][:taken[f]])))
    print("classes with <6 images or topped-up with images a model has seen  (class_id, char, n, n_seen):")
    for s in short:
        print("  ", s)
    print(f"written: {dst}  and  {os.path.join(OUT, 'answer_key.csv')}")


# ------------------------------------------------------------------ eval / verify / score
def acc_table(true_f, preds_by_model, folder_to_cid):
    by = defaultdict(lambda: defaultdict(int))
    for i, t in enumerate(true_f):
        by[t]["n"] += 1
        for m, p in preds_by_model.items():
            by[t][m] += int(p[i] == t)
    return by


def cmd_eval(args):
    pt = load_pt()
    folder_to_cid = pt.BUILTIN_TABLE
    val_c, val_f, all_folder = splits()
    common = sorted(val_c & val_f)
    print(f"CNN2 val {len(val_c)} | FOURTH val {len(val_f)} | unseen by BOTH: {len(common)}")
    true_f = [all_folder[k] for k in common]
    paths = [img_path(k) for k in common]
    models = load_models(pt, MODELS)
    preds = {}
    for n, (m, i2f) in models.items():
        preds[n] = predict(m, i2f, paths, tag=n)
    by = acc_table(true_f, preds, folder_to_cid)
    rows = []
    for folder in sorted(by, key=lambda f: folder_to_cid[f]):
        d = by[folder]
        rows.append([folder_to_cid[folder], folder, label_of(folder), d["n"]] + [d[m] for m in MODELS] +
                    [f"{100 * d[m] / d['n']:.1f}" for m in MODELS])
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "report_per_class.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "folder", "char", "n"] + [f"{m}_correct" for m in MODELS] + [f"{m}_acc%" for m in MODELS])
        w.writerows(rows)
    n = len(common)
    lines = [f"# Held-out accuracy (images NEITHER model trained on): {n} images, {len(by)} of 72 classes present", ""]
    for m in MODELS:
        c = sum(int(a == b) for a, b in zip(preds[m], true_f))
        lines.append(f"- **{m}**: {c}/{n} = {100 * c / n:.2f}%")
    lines += ["", "## Weakest classes per model (only classes with n >= 5; small n is noisy)", ""]
    for m in MODELS:
        worst = sorted((r for r in rows if r[3] >= 5), key=lambda r: float(r[4 + len(MODELS) + list(MODELS).index(m)]))[:8]
        lines.append(f"**{m}**: " + ", ".join(f"{r[0]} {r[2]} ({r[4 + list(MODELS).index(m)]}/{r[3]})" for r in worst))
    lines += ["", "## Classes with too few clean images to judge (n < 5)", ""]
    lines.append(", ".join(f"{r[0]} {r[2]} (n={r[3]})" for r in rows if r[3] < 5) or "none")
    missing = sorted(set(folder_to_cid) - set(by), key=lambda f: folder_to_cid[f])
    lines.append("")
    lines.append("No clean image at all: " + (", ".join(f"{folder_to_cid[f]} {label_of(f)}" for f in missing) or "none"))
    open(os.path.join(OUT, "report.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten: {os.path.join(OUT, 'report_per_class.csv')} , report.md")


def cmd_verify(args):
    pt = load_pt()
    val_c, val_f, all_folder = splits()
    print(f"rebuilt splits: CNN2 val={len(val_c)}  FOURTH val={len(val_f)}  (FOURTH log says 12,545)", flush=True)
    models = load_models(pt, MODELS)
    vf = sorted(val_f)
    true_f = [all_folder[k] for k in vf]
    paths = [img_path(k) for k in vf]
    expect = {"fourth": 12213, "fourth_nofont": 12243}   # exact counts recorded earlier from these checkpoints on the FOURTH val
    for n in ("fourth", "fourth_nofont"):
        m, i2f = models[n]
        p = predict(m, i2f, paths, tag=n)
        c = sum(int(a == b) for a, b in zip(p, true_f))
        print(f"FOURTH val, {n}: {c}/{len(vf)} = {100 * c / len(vf):.2f}%   (recorded earlier: {expect[n]})   {'MATCH' if c == expect[n] else 'DIFFERENT'}", flush=True)
    vc = sorted(val_c)
    m, i2f = models["cnn2"]
    p = predict(m, i2f, [img_path(k) for k in vc], tag="cnn2 val")
    c = sum(int(a == all_folder[k]) for a, k in zip(p, vc))
    print(f"CNN2 val (held out by CNN2): {c}/{len(vc)} = {100 * c / len(vc):.2f}%", flush=True)
    rng = random.Random(1)
    trn = rng.sample(sorted(set(all_folder) - val_c), 4000)
    p = predict(m, i2f, [img_path(k) for k in trn], tag="cnn2 train sample")
    c = sum(int(a == all_folder[k]) for a, k in zip(p, trn))
    print(f"CNN2 on 4000 of ITS OWN TRAIN images: {c}/4000 = {100 * c / 4000:.2f}%  (should be higher than on val if the split is right)", flush=True)


def read_predictions(path):
    """Accept the values file (one class_id per line) or predictions_<GROUP>.csv (gt_path,<GROUP>)."""
    text = open(path, encoding="utf-8").read()
    if path.lower().endswith(".csv"):
        rows = list(csv.reader(io.StringIO(text)))[1:]
        return [r[-1].strip() for r in rows if r]
    return text.split()


def cmd_score(args):
    key_path = KEY or os.path.join(OUT, "answer_key.csv")
    key = list(csv.DictReader(open(key_path, encoding="utf-8")))
    preds = read_predictions(args.file)
    assert len(preds) == len(key), f"{len(preds)} predictions vs {len(key)} rows in {key_path}"
    ok = [p == r["gt_class_id"] for p, r in zip(preds, key)]
    print(f"overall: {sum(ok)}/{len(ok)} = {100 * sum(ok) / len(ok):.2f}%")
    for tb in ("none", "fourth", "cnn2", "both"):
        idx = [i for i, r in enumerate(key) if r["trained_by"] == tb]
        if idx:
            print(f"  trained_by={tb:7s}: {sum(ok[i] for i in idx)}/{len(idx)}" + ("   <- clean rows (neither model trained on them)" if tb == "none" else ""))
    print("wrong rows (row, file, truth -> predicted):")
    for i, (o, r) in enumerate(zip(ok, key)):
        if not o:
            print(f"  {i + 1:3d} {r['gt_path']}  {r['gt_class_id']} -> {preds[i]}   [{r['trained_by']}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Rehearsal tools (options go BEFORE the command)")
    ap.add_argument("--cnn2", help="CNN2 repo folder (has dataset/round2 and model.pt)")
    ap.add_argument("--fourth", help="CNN_font_and_nofont repo folder (has dataset/round2, model.pt, model_nofont.pt)")
    ap.add_argument("--out", help="output folder for build/eval (default: <this script's folder>/rehearsal)")
    ap.add_argument("--key", help="answer_key.csv for `score` (default: <out>/answer_key.csv)")
    ap.add_argument("--predict-script", help="path of predict_test_embed.py (default: next to this script)")
    ap.add_argument("--model", action="append", help="NAME=PATH, replace/add a checkpoint (repeatable)")
    ap.add_argument("--seed", type=int, default=42, help="sampling seed for `build` (does NOT change which images count as unseen)")
    ap.add_argument("--n", type=int, default=500, help="rehearsal set size for `build`")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build").set_defaults(fn=cmd_build)
    sub.add_parser("eval").set_defaults(fn=cmd_eval)
    sub.add_parser("verify").set_defaults(fn=cmd_verify)
    sc = sub.add_parser("score")
    sc.add_argument("file")
    sc.set_defaults(fn=cmd_score)
    a = ap.parse_args()
    configure(a)
    a.fn(a)
