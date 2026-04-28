"""
Evaluate Student B's anonymisation on Student A's FaceRecognizer.

Metrics computed:
- ID Similarity Drop: cosine similarity between original image
  embedding and its anonymised counterpart.
  (Lower = better anonymisation.)
- Impostor baseline: cosine similarity between pairs of different
  original identities (random impostor pairs).
- Verification decision at threshold t=0.5:
  if the anonymiser is strong, an ID recognition system that thinks
  <cos >= t> implies "same person" should now say "different person"
  (i.e. the verifier cannot authenticate the original identity).
- Also reports identification accuracy on the anonymised set: what
  fraction of anonymised images get classified as the SAME Pins
  identity as the original image (upper-bound leakage proxy).

Inputs:
  original_dir:   folder of original CelebA-HQ images (e.g. 10000.jpg)
  anonymised_dir: folder of Student B's output (e.g. 10000_anonymized.jpg)
  pattern:        filename join-key convention. We match by the
                  numeric stem: 10000.jpg <-> 10000_anonymized.jpg

Saves:
  results/anonymisation_evaluation.json
  results/anonymisation_histogram.png
"""

import argparse
import json
import os
import random
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

import config
from inference import FaceRecognizer


ORIGINAL_DIR = "/Users/yuchendeng/Desktop/comp4026/comp4026/Untitled Folder"
ANON_DIR = "/Users/yuchendeng/Desktop/comp4026/comp4026/batch"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)

THRESHOLD = 0.5  # face verification threshold used by FaceRecognizer.verify


def stem_key(filename: str) -> str:
    """Strip extension and any _anonymized suffix to get a join-key."""
    name = os.path.splitext(filename)[0]
    name = re.sub(r"_anonymized$", "", name, flags=re.IGNORECASE)
    return name


def list_images(folder: str) -> dict:
    """Return {stem: full_path} for a folder of images."""
    out = {}
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            out[stem_key(f)] = os.path.join(folder, f)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", default=ORIGINAL_DIR)
    parser.add_argument("--anonymised", default=ANON_DIR)
    parser.add_argument("--n_impostor", type=int, default=500)
    args = parser.parse_args()

    random.seed(42)

    print("=" * 60)
    print("Anonymisation Evaluation (Student A + Student B joint eval)")
    print("=" * 60)
    print(f"Original dir : {args.original}")
    print(f"Anonymised dir: {args.anonymised}")

    originals = list_images(args.original)
    anonymised = list_images(args.anonymised)

    # Match by stem
    common = sorted(set(originals) & set(anonymised))
    print(f"Paired images: {len(common)}")
    if not common:
        raise SystemExit("No paired images found. Check filename convention.")

    # Load recogniser
    fr = FaceRecognizer()

    # ------------------------------------------------------------------
    # 1) Compute embeddings for originals + anonymised
    # ------------------------------------------------------------------
    orig_embs, anon_embs, keys = [], [], []
    print("\n[1/3] Extracting embeddings ...")
    for k in tqdm(common):
        try:
            e_o = fr.get_embedding(originals[k])
            e_a = fr.get_embedding(anonymised[k])
            orig_embs.append(e_o)
            anon_embs.append(e_a)
            keys.append(k)
        except Exception as e:
            print(f"  skip {k}: {e}")
    orig_embs = np.stack(orig_embs)
    anon_embs = np.stack(anon_embs)
    N = len(keys)
    print(f"  embeddings: {N} pairs")

    # ------------------------------------------------------------------
    # 2) ID similarity: original <-> its anonymised
    # ------------------------------------------------------------------
    # embeddings are L2-normalised -> dot product = cosine
    genuine = (orig_embs * anon_embs).sum(axis=1)
    genuine_mean = float(genuine.mean())
    genuine_std = float(genuine.std())

    # ------------------------------------------------------------------
    # 3) Impostor baseline: random pairs of DIFFERENT original identities
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)
    impostor = []
    for _ in range(args.n_impostor):
        i, j = rng.choice(N, size=2, replace=False)
        impostor.append(float(orig_embs[i] @ orig_embs[j]))
    impostor = np.array(impostor)

    # Sanity: original <-> original (self) should be 1.0 for every row
    self_sim = (orig_embs * orig_embs).sum(axis=1)

    # ------------------------------------------------------------------
    # 4) Verification outcome at threshold 0.5
    # ------------------------------------------------------------------
    # Question: "Does the recogniser still think the anonymised image is
    # the same person as the original?"
    # If cos >= threshold we count it as "authentication succeeded"
    # (i.e. privacy BROKEN for that image).
    authenticated = int((genuine >= THRESHOLD).sum())
    auth_rate = authenticated / N  # lower is better
    privacy_protection = 1.0 - auth_rate

    # "Impostor" pair authentication rate at same threshold (for context)
    impostor_auth_rate = float((impostor >= THRESHOLD).mean())

    # ------------------------------------------------------------------
    # 5) Closed-set ID leakage on anonymised images
    #    (share of anonymised images that fr.predict() maps to the
    #    same Pins identity as the original image does)
    # ------------------------------------------------------------------
    print("\n[2/3] Closed-set ID leakage on anonymised images ...")
    orig_pred, anon_pred = [], []
    for k in tqdm(keys):
        orig_pred.append(fr.predict(originals[k])["identity"])
        anon_pred.append(fr.predict(anonymised[k])["identity"])
    id_match = [a == b for a, b in zip(orig_pred, anon_pred)]
    id_match_rate = float(np.mean(id_match))

    # ------------------------------------------------------------------
    # 6) Plots
    # ------------------------------------------------------------------
    print("\n[3/3] Plotting distributions ...")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.hist(impostor, bins=50, alpha=.55, label=f"Impostor pairs (different ID)  n={len(impostor)}",
            color="#9ca3af", edgecolor="white")
    ax.hist(genuine, bins=50, alpha=.75, label=f"Original ↔ Anonymised  n={N}",
            color="#3bb6f5", edgecolor="white")
    ax.axvline(THRESHOLD, color="#ef4444", ls="--", lw=1.4,
               label=f"Verification threshold (t={THRESHOLD})")
    ax.set_xlabel("Cosine similarity of FaceRecognizer embeddings")
    ax.set_ylabel("# pairs")
    ax.set_title("Anonymisation effect on face verification")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, "anonymisation_histogram.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  saved {fig_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary = {
        "n_pairs": N,
        "n_impostor_pairs": len(impostor),
        "threshold": THRESHOLD,
        "genuine_cosine_mean": genuine_mean,
        "genuine_cosine_std": genuine_std,
        "genuine_cosine_median": float(np.median(genuine)),
        "genuine_cosine_p05": float(np.percentile(genuine, 5)),
        "genuine_cosine_p95": float(np.percentile(genuine, 95)),
        "impostor_cosine_mean": float(impostor.mean()),
        "impostor_cosine_std": float(impostor.std()),
        "self_similarity_mean": float(self_sim.mean()),

        "authentication_success_on_anonymised": auth_rate,
        "privacy_protection_rate": privacy_protection,
        "impostor_auth_rate_at_same_threshold": impostor_auth_rate,

        "closed_set_id_leakage_rate": id_match_rate,
        "note": (
            "Original dir = real CelebA-HQ images, "
            "Anonymised dir = Student B DDPM outputs. "
            "FaceRecognizer is trained on Pins (105 celebrities) "
            "and reused here as a discriminator."
        ),
    }
    out_json = os.path.join(OUT_DIR, "anonymisation_evaluation.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  saved {out_json}")

    # Human-readable printout
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Paired images                  : {N}")
    print(f"Genuine  cos  (orig↔anon)      : {genuine_mean:.4f} ± {genuine_std:.4f}")
    print(f"Impostor cos  (orig↔otherOrig) : {impostor.mean():.4f} ± {impostor.std():.4f}")
    print(f"--- at threshold t = {THRESHOLD} ---")
    print(f"Auth. succeeded on anonymised  : {auth_rate*100:.2f}%  (privacy leaked)")
    print(f"  => Privacy Protection Rate   : {privacy_protection*100:.2f}%")
    print(f"Impostor auth rate (reference) : {impostor_auth_rate*100:.2f}%")
    print(f"Closed-set ID leakage (top-1)  : {id_match_rate*100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
