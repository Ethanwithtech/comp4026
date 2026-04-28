"""
Prepare a sample of Pins test-set images for Student B's anonymisation model.

Selects N images per identity from the test split (seed=42, matching the
training pipeline), resizes them to 256x256 (CelebA-HQ-compatible), and
packages them with a labels.csv and a README for Student B.

Usage:
    python3 prepare_pins_for_anon.py            # default: 3 images per identity
    python3 prepare_pins_for_anon.py --n 5      # 5 per identity
"""

import argparse
import csv
import os
import shutil
import sys
import zipfile

from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from utils.dataset import discover_dataset, download_dataset, split_dataset  # noqa: E402

OUTPUT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "pins_for_anon",
)
IMAGES_DIR = os.path.join(OUTPUT_ROOT, "images")
LABELS_FILE = os.path.join(OUTPUT_ROOT, "labels.csv")
README_FILE = os.path.join(OUTPUT_ROOT, "README.txt")
ZIP_FILE = os.path.join(os.path.dirname(OUTPUT_ROOT), "pins_for_anon.zip")

TARGET_SIZE = 256  # Matches CelebA-HQ crop size used by Student B's DDPM


def sanitize(name: str) -> str:
    """Drop the 'pins_' prefix and replace spaces with underscores."""
    base = name[len("pins_"):] if name.lower().startswith("pins_") else name
    return base.strip().replace(" ", "_")


def center_resize(img: Image.Image, size: int = TARGET_SIZE) -> Image.Image:
    """Resize the shorter side to `size` then center-crop to size x size."""
    img = img.convert("RGB")
    w, h = img.size
    scale = size / min(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img = img.resize((nw, nh), Image.LANCZOS)
    # center crop
    left = (nw - size) // 2
    top = (nh - size) // 2
    return img.crop((left, top, left + size, top + size))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3,
                        help="Images per identity (default: 3)")
    parser.add_argument("--no-zip", action="store_true",
                        help="Skip creating the .zip archive")
    args = parser.parse_args()

    # Clean output dirs
    if os.path.isdir(OUTPUT_ROOT):
        shutil.rmtree(OUTPUT_ROOT)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # Use the exact same test split the training pipeline used
    data_path = download_dataset()
    identity_images, class_names = discover_dataset(data_path)
    _, _, test_data = split_dataset(identity_images, class_names, seed=42)

    # Group test items by identity label
    label_to_name = {i: sanitize(name) for i, name in enumerate(class_names)}
    buckets: dict[int, list[str]] = {i: [] for i in range(len(class_names))}
    for path, label in test_data:
        buckets[label].append(path)

    rows: list[tuple[str, str, str]] = []  # (filename, identity, src_path)
    skipped: list[str] = []

    for label in sorted(buckets.keys()):
        name = label_to_name[label]
        paths = buckets[label]
        if not paths:
            skipped.append(name)
            continue
        # Deterministic selection: first N from the (already-shuffled-at-split) list
        picks = paths[: args.n]
        for i, src in enumerate(picks):
            dst_name = f"{name}_{i:02d}.jpg"
            try:
                img = Image.open(src)
                img = center_resize(img, TARGET_SIZE)
                img.save(os.path.join(IMAGES_DIR, dst_name), quality=95)
                rows.append((dst_name, name, src))
            except Exception as e:
                print(f"[skip] {src}: {e}")

    # labels.csv
    with open(LABELS_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "identity"])
        for fn, ident, _ in rows:
            w.writerow([fn, ident])

    # README for Student B
    n_ids = len({r[1] for r in rows})
    with open(README_FILE, "w", encoding="utf-8") as f:
        f.write(
            f"""Pins sample for anonymisation — COMP4026 Group 04
========================================================

Contents
--------
- images/                 {len(rows)} face images (256x256 JPG) from {n_ids} identities
- labels.csv              filename -> ground-truth identity

What to do
----------
1. Run your anonymisation model on every image in images/.
2. Save the output with the SAME FILENAME into a new folder named
   anonymised/ (e.g. images/Leonardo_DiCaprio_00.jpg
   -> anonymised/Leonardo_DiCaprio_00.jpg).
3. Keep the 256x256 JPG format (same as your CelebA-HQ outputs).
4. Send back the anonymised/ folder (zipped).

Notes
-----
- These are Pins Face Recognition test-set images (seed=42 split),
  so Student A's FaceRecognizer was originally trained on other
  images of the SAME identities. This lets us measure
  "Accuracy on anonymised images" cleanly, because the classifier
  is supposed to recognise these people.
- No identity labels are needed for your model to run — it only
  reads the image files. labels.csv is for Student A's evaluation.
"""
        )

    # Zip it
    if not args.no_zip:
        if os.path.exists(ZIP_FILE):
            os.remove(ZIP_FILE)
        with zipfile.ZipFile(ZIP_FILE, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(OUTPUT_ROOT):
                for fn in files:
                    full = os.path.join(root, fn)
                    arc = os.path.relpath(full, os.path.dirname(OUTPUT_ROOT))
                    zf.write(full, arc)

    print("\n==== DONE ====")
    print(f"Identities covered     : {n_ids} / {len(class_names)}")
    print(f"Images written         : {len(rows)}")
    if skipped:
        print(f"Identities with no test images: {len(skipped)}")
    print(f"Output folder          : {OUTPUT_ROOT}")
    print(f"labels.csv             : {LABELS_FILE}")
    print(f"README                 : {README_FILE}")
    if not args.no_zip:
        sz = os.path.getsize(ZIP_FILE) / (1024 * 1024)
        print(f"Zip archive            : {ZIP_FILE}  ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
