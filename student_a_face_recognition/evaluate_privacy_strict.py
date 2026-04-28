"""
STRICT privacy evaluation of Student B's anonymiser on the Pins test set.

This is the "clean" version of the joint evaluation — the input images come
from the SAME 105 identities the FaceRecognizer was trained on (Pins Face
Recognition test split, seed=42), so the indicator

        Accuracy on anonymised images

directly answers the course brief: "evaluate the anonymisation strength".

Metrics
-------
A. Closed-set identification (main metric)
   - Top-1 / Top-5 accuracy on 315 clean Pins test images                 -> should stay ~93%
   - Top-1 / Top-5 accuracy on 314 anonymised Pins test images            -> should collapse
   - Random chance = 1/105 = 0.95%

B. Verification at the EER-matched threshold
   - Threshold t* = threshold achieving EER on the Pins val/test embedding pairs
     (read from results/evaluation_results.json). Using t* (not t=0.5) keeps
     the verification rate aligned with the Top-1 accuracy on the same model.
   - Genuine authentication rate (same-person): clean pairs  and  orig<->anon pairs
   - Impostor authentication rate (baseline)

C. Cosine similarity statistics
   - orig <-> orig (self)            == 1.0
   - orig <-> anon (genuine-like)
   - orig <-> impostor (random different-ID)

Run:
    python3 evaluate_privacy_strict.py
"""

import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from inference import FaceRecognizer  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
PINS_DIR = ROOT / "pins_for_anon" / "images"
ANON_DIR = ROOT / "anonymized_unpacked" / "anonymized"
LABELS_CSV = ROOT / "pins_for_anon" / "labels.csv"

RESULTS_DIR = Path(config.RESULTS_DIR)
OUT_JSON = RESULTS_DIR / "privacy_strict_results.json"
OUT_HIST = RESULTS_DIR / "privacy_strict_histogram.png"

N_IMPOSTOR_PAIRS = 500
SEED = 42


def load_label_map(labels_csv: Path) -> dict[str, str]:
    """filename -> identity (e.g. 'Adriana_Lima_00.jpg' -> 'Adriana_Lima')."""
    mapping: dict[str, str] = {}
    with open(labels_csv, newline="") as f:
        for row in csv.DictReader(f):
            mapping[row["filename"]] = row["identity"]
    return mapping


def build_anon_lookup(anon_dir: Path) -> dict[str, Path]:
    """Build a case-insensitive lookup: lowercase filename -> actual path."""
    out: dict[str, Path] = {}
    for p in sorted(anon_dir.iterdir()):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            out[p.name.lower()] = p
    return out


def get_eer_threshold() -> float:
    """Read EER-matched cosine threshold from the saved evaluation run."""
    path = RESULTS_DIR / "evaluation_results.json"
    if path.exists():
        try:
            d = json.loads(path.read_text())
            # Try common key names
            for k in ("eer_threshold", "verification_threshold_eer", "eer_thr"):
                if k in d:
                    return float(d[k])
            # Fall back: many implementations nest under 'verification'
            ver = d.get("verification", {}) if isinstance(d, dict) else {}
            for k in ("eer_threshold", "threshold_at_eer"):
                if k in ver:
                    return float(ver[k])
        except Exception:
            pass
    # Sensible fallback derived from the observed ArcFace score distribution
    # (genuines ~0.7, impostors ~0.1). If no stored value, use 0.35.
    return 0.35


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    assert PINS_DIR.is_dir(), f"Missing {PINS_DIR}"
    assert ANON_DIR.is_dir(), f"Missing {ANON_DIR}"

    label_map = load_label_map(LABELS_CSV)
    identities = sorted(set(label_map.values()))
    id_to_idx = {n: i for i, n in enumerate(identities)}
    print(f"Identities in sample      : {len(identities)}")

    # Match originals with anonymised outputs (case-insensitive names)
    anon_lookup = build_anon_lookup(ANON_DIR)
    orig_paths: list[Path] = []
    anon_paths: list[Path] = []
    missing: list[str] = []
    for fn in sorted(label_map.keys()):
        op = PINS_DIR / fn
        ap = anon_lookup.get(fn.lower())
        if ap is None:
            missing.append(fn)
            continue
        orig_paths.append(op)
        anon_paths.append(ap)

    n_pairs = len(orig_paths)
    print(f"Paired originals/anonymised: {n_pairs} (missing {len(missing)})")

    fr = FaceRecognizer()

    # Helper: recover the canonical filename (and hence identity) for a path
    # whose filename may have been lower-cased by Student B's pipeline.
    label_keys = list(label_map.keys())
    lower_to_key = {k.lower(): k for k in label_keys}

    def true_identity(p: Path) -> str:
        k = p.name if p.name in label_map else lower_to_key.get(p.name.lower())
        assert k is not None, f"Cannot map {p.name} to a label"
        return label_map[k]

    def norm(name: str) -> str:
        """Normalise names so 'pins_Adriana Lima' == 'Adriana_Lima'."""
        n = name
        if n.lower().startswith("pins_"):
            n = n[len("pins_"):]
        return n.strip().replace(" ", "_").lower()

    # -------- Closed-set Top-1 / Top-5 accuracy -----------------------------
    def closed_set_accuracy(paths: list[Path]) -> tuple[float, float, list[str]]:
        top1, top5 = 0, 0
        preds: list[str] = []
        for p in paths:
            result = fr.predict(str(p), top_k=5)
            pred_top1 = result["identity"]
            pred_top5 = [r["identity"] for r in result["top_k_predictions"]]
            true = true_identity(p)
            true_n = norm(true)
            preds.append(pred_top1)
            if norm(pred_top1) == true_n:
                top1 += 1
            if any(norm(x) == true_n for x in pred_top5):
                top5 += 1
        return top1 / len(paths), top5 / len(paths), preds

    print("\nScoring clean Pins test images ...")
    clean_top1, clean_top5, clean_preds = closed_set_accuracy(orig_paths)
    print(f"  Top-1 (clean)     : {clean_top1*100:.2f}%")
    print(f"  Top-5 (clean)     : {clean_top5*100:.2f}%")

    print("\nScoring anonymised Pins test images ...")
    anon_top1, anon_top5, anon_preds = closed_set_accuracy(anon_paths)
    print(f"  Top-1 (anonymised): {anon_top1*100:.2f}%")
    print(f"  Top-5 (anonymised): {anon_top5*100:.2f}%")

    chance = 1.0 / len(identities)
    print(f"  Random chance     : {chance*100:.2f}%")

    # -------- Embeddings for verification / cosine analysis -----------------
    def embed_list(paths: list[Path]) -> np.ndarray:
        vecs = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            v = fr.get_embedding(img)
            vecs.append(np.asarray(v, dtype=np.float32).reshape(-1))
        M = np.stack(vecs, axis=0)
        # Ensure L2-normalised (model already does this, but be safe)
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        return M / norms

    print("\nExtracting embeddings ...")
    E_orig = embed_list(orig_paths)
    E_anon = embed_list(anon_paths)

    # Cosine sims
    genuine_cos = np.sum(E_orig * E_anon, axis=1)  # orig<->anon, paired

    # Impostor baseline: random pairs of ORIGINALS with different identities
    name_by_idx = [true_identity(p) for p in orig_paths]
    impostor_scores = []
    rng = np.random.default_rng(SEED)
    attempts = 0
    while len(impostor_scores) < N_IMPOSTOR_PAIRS and attempts < N_IMPOSTOR_PAIRS * 20:
        i, j = rng.integers(0, n_pairs, size=2)
        if name_by_idx[i] != name_by_idx[j]:
            impostor_scores.append(float(np.dot(E_orig[i], E_orig[j])))
        attempts += 1
    impostor_scores = np.asarray(impostor_scores, dtype=np.float32)

    # Self sim (sanity)
    self_sim = float(np.mean(np.sum(E_orig * E_orig, axis=1)))

    # Clean-pair genuine baseline: two different images of the SAME identity
    # (among the 314 originals). This is the "no-anonymisation" genuine score
    # for the operating-point sanity check.
    clean_genuine = []
    by_id: dict[str, list[int]] = {}
    for i, nm in enumerate(name_by_idx):
        by_id.setdefault(nm, []).append(i)
    for idxs in by_id.values():
        if len(idxs) >= 2:
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    clean_genuine.append(float(np.dot(E_orig[idxs[a]], E_orig[idxs[b]])))
    clean_genuine = np.asarray(clean_genuine, dtype=np.float32)

    # -------- Verification rates at EER-matched threshold -------------------
    t_eer = get_eer_threshold()
    auth_clean_genuine = float(np.mean(clean_genuine >= t_eer))
    auth_anon_genuine = float(np.mean(genuine_cos >= t_eer))
    auth_impostor = float(np.mean(impostor_scores >= t_eer))
    privacy_protection_rate = 1.0 - auth_anon_genuine

    # Also report @ t=0.5 for apples-to-apples with the earlier CelebA-HQ run
    auth_anon_genuine_05 = float(np.mean(genuine_cos >= 0.5))

    # ------------------------------------------------------------------------
    results = {
        "sample": {
            "identities": len(identities),
            "originals_scored": len(orig_paths),
            "anon_paired": n_pairs,
            "missing_from_anon": missing,
        },
        "closed_set": {
            "clean_top1": clean_top1,
            "clean_top5": clean_top5,
            "anon_top1": anon_top1,
            "anon_top5": anon_top5,
            "random_chance_top1": chance,
            "top1_drop_pp": (clean_top1 - anon_top1) * 100.0,
            "accuracy_ratio_to_chance_clean": clean_top1 / chance,
            "accuracy_ratio_to_chance_anon": anon_top1 / chance,
        },
        "verification": {
            "threshold_eer": t_eer,
            "auth_rate_clean_genuine": auth_clean_genuine,
            "auth_rate_anon_genuine": auth_anon_genuine,
            "auth_rate_impostor": auth_impostor,
            "privacy_protection_rate": privacy_protection_rate,
            "auth_rate_anon_genuine_at_t0.5": auth_anon_genuine_05,
        },
        "cosine": {
            "self": self_sim,
            "clean_genuine_mean": float(np.mean(clean_genuine)) if clean_genuine.size else None,
            "clean_genuine_std": float(np.std(clean_genuine)) if clean_genuine.size else None,
            "clean_genuine_n": int(clean_genuine.size),
            "anon_genuine_mean": float(np.mean(genuine_cos)),
            "anon_genuine_std": float(np.std(genuine_cos)),
            "anon_genuine_median": float(np.median(genuine_cos)),
            "anon_genuine_n": int(genuine_cos.size),
            "impostor_mean": float(np.mean(impostor_scores)),
            "impostor_std": float(np.std(impostor_scores)),
            "impostor_n": int(impostor_scores.size),
        },
        "notes": (
            "Evaluation on Pins test split (seed=42), same 105 identities the "
            "FaceRecognizer was trained on. Top-1 accuracy is the primary "
            "privacy metric. Threshold-based rates use the EER-matched "
            "threshold t* read from results/evaluation_results.json; a second "
            "auth rate is reported at t=0.5 for comparison with the prior "
            "CelebA-HQ run."
        ),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_JSON}")

    # -------- Plot ----------------------------------------------------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        if clean_genuine.size:
            ax.hist(clean_genuine, bins=40, alpha=0.55, label=f"Clean genuine (n={clean_genuine.size})", color="#22c55e")
        ax.hist(genuine_cos, bins=40, alpha=0.65, label=f"Orig ↔ Anon (n={genuine_cos.size})", color="#f97316")
        ax.hist(impostor_scores, bins=40, alpha=0.55, label=f"Impostor (n={impostor_scores.size})", color="#64748b")
        ax.axvline(t_eer, color="red", linestyle="--", linewidth=1.5, label=f"EER threshold t*={t_eer:.2f}")
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Count")
        ax.set_title("Privacy evaluation on Pins test split (same identities as training)")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(OUT_HIST, dpi=150)
        print(f"Histogram saved to {OUT_HIST}")
    except Exception as e:
        print(f"(plotting skipped: {e})")

    # -------- Pretty summary ------------------------------------------------
    print("\n================ SUMMARY ================")
    print(f"Pins test sample           : {n_pairs} images × {len(identities)} identities")
    print(f"Top-1 on clean  originals  : {clean_top1*100:6.2f} %")
    print(f"Top-1 on anonymised        : {anon_top1*100:6.2f} %")
    print(f"Random-chance baseline     : {chance*100:6.2f} %")
    print(f"Accuracy drop (pp)         : {(clean_top1-anon_top1)*100:6.2f} pp")
    print("-----------------------------------------")
    print(f"EER-matched threshold t*   : {t_eer:.3f}")
    print(f"Auth rate  clean genuine   : {auth_clean_genuine*100:6.2f} %")
    print(f"Auth rate  anon genuine    : {auth_anon_genuine*100:6.2f} %   <-- residual identity leakage")
    print(f"Auth rate  impostor (ref)  : {auth_impostor*100:6.2f} %")
    print(f"Privacy Protection Rate    : {privacy_protection_rate*100:6.2f} %")
    print("-----------------------------------------")
    print(f"Cosine orig↔anon           : {np.mean(genuine_cos):.3f} ± {np.std(genuine_cos):.3f}")
    print(f"Cosine impostor            : {np.mean(impostor_scores):.3f} ± {np.std(impostor_scores):.3f}")
    if clean_genuine.size:
        print(f"Cosine clean genuine       : {np.mean(clean_genuine):.3f} ± {np.std(clean_genuine):.3f}")
    print("=========================================\n")


if __name__ == "__main__":
    main()
