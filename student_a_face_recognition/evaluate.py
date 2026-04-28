"""
Evaluation script for Face Recognition Model (Student A).
COMP4026 Group Project: Anonymised Facial Expression Recognition

This script evaluates:
1. Face Recognition Accuracy on ORIGINAL test images
   -> Validates the reliability of the ID classifier.
2. Face Recognition Accuracy on ANONYMISED images
   -> Evaluates the anonymisation strength.
   -> Lower accuracy = stronger privacy protection.

Usage:
    # Evaluate on original test set
    python evaluate.py --data_path PATH_TO_DATASET

    # Evaluate on anonymised images
    python evaluate.py --anon_dir PATH_TO_ANONYMISED_DIR --label_file labels.csv

    # Both evaluations
    python evaluate.py --data_path PATH --anon_dir ANON_PATH --label_file labels.csv
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from model import FaceRecognitionModel
from utils.dataset import (
    AnonymisedFaceDataset,
    discover_dataset,
    get_dataloaders,
    get_transforms,
    split_dataset,
    FaceDataset,
)


def load_model(checkpoint_path: str = None) -> tuple:
    """Load the trained face recognition model."""
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.MODEL_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}. "
            "Please train the model first using train.py"
        )

    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)

    num_classes = checkpoint['num_classes']
    class_names = checkpoint['class_names']
    model_config = checkpoint.get('config', {})

    model = FaceRecognitionModel(
        num_classes=num_classes,
        embedding_dim=model_config.get('embedding_dim', config.EMBEDDING_DIM),
        backbone=model_config.get('backbone', config.BACKBONE),
        pretrained=False,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.DEVICE)
    model.eval()

    print(f"Model loaded: {num_classes} classes, "
          f"best val acc: {checkpoint.get('best_val_acc', 'N/A')}")
    return model, class_names


@torch.no_grad()
def evaluate_identification(model, data_loader, num_classes, desc="Evaluating"):
    """
    Evaluate face IDENTIFICATION accuracy (closed-set).
    The model predicts which identity the face belongs to.

    Returns:
        results dict with accuracy, top-k accuracy, per-class accuracy, etc.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    pbar = tqdm(data_loader, desc=desc)
    for images, labels in pbar:
        images = images.to(config.DEVICE)
        logits = model(images)
        probs = F.softmax(logits, dim=1)

        _, predicted = logits.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Overall accuracy
    accuracy = accuracy_score(all_labels, all_preds) * 100

    # Top-K accuracy
    topk_results = {}
    for k in config.TOP_K:
        if k <= num_classes:
            topk_acc = top_k_accuracy_score(
                all_labels, all_probs, k=k, labels=range(num_classes)
            ) * 100
            topk_results[f"top{k}_accuracy"] = topk_acc

    # Per-class accuracy
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    for pred, label in zip(all_preds, all_labels):
        per_class_total[label] += 1
        if pred == label:
            per_class_correct[label] += 1

    per_class_acc = {}
    for cls in per_class_total:
        per_class_acc[int(cls)] = (
            per_class_correct[cls] / per_class_total[cls] * 100
            if per_class_total[cls] > 0 else 0.0
        )

    return {
        'accuracy': accuracy,
        'topk_accuracy': topk_results,
        'per_class_accuracy': per_class_acc,
        'all_preds': all_preds,
        'all_labels': all_labels,
        'all_probs': all_probs,
        'num_samples': len(all_labels),
    }


@torch.no_grad()
def evaluate_verification(model, data_loader, desc="Verification"):
    """
    Extract embeddings for face VERIFICATION evaluation.
    Used to compute cosine similarity between pairs.

    Returns:
        embeddings: (N, embedding_dim) numpy array
        labels: (N,) numpy array
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    pbar = tqdm(data_loader, desc=desc)
    for images, labels in pbar:
        images = images.to(config.DEVICE)
        embeddings = model.get_embedding(images)
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.extend(labels.numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.array(all_labels)

    return all_embeddings, all_labels


def compute_verification_metrics(embeddings, labels, num_pairs=10000):
    """
    Compute face verification metrics (genuine/impostor pairs).

    Returns:
        dict with TAR@FAR, EER, etc.
    """
    np.random.seed(42)
    n = len(labels)

    # Generate genuine pairs (same identity)
    genuine_scores = []
    impostor_scores = []

    label_to_indices = defaultdict(list)
    for i, l in enumerate(labels):
        label_to_indices[l].append(i)

    # Genuine pairs
    count = 0
    for label, indices in label_to_indices.items():
        if len(indices) >= 2:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    sim = np.dot(embeddings[indices[i]], embeddings[indices[j]])
                    genuine_scores.append(sim)
                    count += 1
                    if count >= num_pairs // 2:
                        break
                if count >= num_pairs // 2:
                    break
        if count >= num_pairs // 2:
            break

    # Impostor pairs
    all_labels_unique = list(label_to_indices.keys())
    count = 0
    while count < num_pairs // 2:
        l1, l2 = np.random.choice(all_labels_unique, 2, replace=False)
        i1 = np.random.choice(label_to_indices[l1])
        i2 = np.random.choice(label_to_indices[l2])
        sim = np.dot(embeddings[i1], embeddings[i2])
        impostor_scores.append(sim)
        count += 1

    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    # Compute TAR@FAR for various thresholds
    thresholds = np.linspace(-1, 1, 1000)
    tar_list = []
    far_list = []
    for t in thresholds:
        tar = np.mean(genuine_scores >= t)
        far = np.mean(impostor_scores >= t)
        tar_list.append(tar)
        far_list.append(far)

    tar_arr = np.array(tar_list)
    far_arr = np.array(far_list)

    # Find EER
    diffs = np.abs(tar_arr - (1 - far_arr))
    eer_idx = np.argmin(diffs)
    eer = far_arr[eer_idx]

    # TAR @ FAR = 0.01, 0.001
    tar_at_far = {}
    for target_far in [0.1, 0.01, 0.001]:
        idx = np.argmin(np.abs(far_arr - target_far))
        tar_at_far[f"TAR@FAR={target_far}"] = tar_arr[idx]

    return {
        'eer': float(eer),
        'tar_at_far': tar_at_far,
        'genuine_scores': genuine_scores.tolist(),
        'impostor_scores': impostor_scores.tolist(),
        'thresholds': thresholds.tolist(),
        'tar': tar_arr.tolist(),
        'far': far_arr.tolist(),
    }


@torch.no_grad()
def evaluate_anonymised(model, anon_dir, label_file, class_names):
    """
    Evaluate face recognition on ANONYMISED images.
    This measures the privacy protection strength.

    A LOWER accuracy means STRONGER anonymisation.

    Args:
        model: Trained face recognition model
        anon_dir: Directory with anonymised images
        label_file: CSV mapping anonymised images to original identities
        class_names: List of identity class names

    Returns:
        dict with accuracy and analysis
    """
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}

    anon_dataset = AnonymisedFaceDataset(
        image_dir=anon_dir,
        label_file=label_file,
        transform=get_transforms(is_training=False),
    )

    anon_loader = DataLoader(
        anon_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    model.eval()
    all_preds = []
    all_true_labels = []
    all_paths = []

    for images, labels, paths in tqdm(anon_loader, desc="Evaluating anonymised"):
        images = images.to(config.DEVICE)
        logits = model(images)
        _, predicted = logits.max(1)

        all_preds.extend(predicted.cpu().numpy())
        # Convert string labels to indices
        true_indices = []
        for l in labels:
            if isinstance(l, str) and l in name_to_idx:
                true_indices.append(name_to_idx[l])
            elif isinstance(l, (int, np.integer)):
                true_indices.append(int(l))
            else:
                true_indices.append(-1)

        all_true_labels.extend(true_indices)
        all_paths.extend(paths)

    all_preds = np.array(all_preds)
    all_true_labels = np.array(all_true_labels)

    # Filter out unlabelled samples
    valid_mask = all_true_labels >= 0
    if valid_mask.sum() > 0:
        valid_preds = all_preds[valid_mask]
        valid_labels = all_true_labels[valid_mask]
        accuracy = accuracy_score(valid_labels, valid_preds) * 100

        # How many identities were correctly recognized despite anonymisation
        correctly_identified = (valid_preds == valid_labels).sum()
        total_valid = len(valid_labels)
    else:
        accuracy = None
        correctly_identified = 0
        total_valid = 0

    return {
        'anonymised_accuracy': accuracy,
        'correctly_identified': int(correctly_identified),
        'total_images': int(total_valid),
        'privacy_protection_rate': (
            100 - accuracy if accuracy is not None else None
        ),
        'note': (
            "Lower accuracy = stronger anonymisation. "
            "Ideally, accuracy should drop to near random chance "
            f"({100/len(class_names):.2f}%) for perfect anonymisation."
        ),
    }


def plot_training_history(history_path, save_dir):
    """Plot training curves."""
    with open(history_path, "r") as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', color='#2196F3')
    axes[0].plot(history['val_loss'], label='Val Loss', color='#F44336')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc', color='#2196F3')
    axes[1].plot(history['val_acc'], label='Val Acc', color='#F44336')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Learning rate
    axes[2].plot(history['lr'], color='#4CAF50')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_confusion_matrix(all_labels, all_preds, class_names, save_dir,
                          title="Confusion Matrix", max_classes=30):
    """Plot confusion matrix (for manageable number of classes)."""
    num_classes = len(set(all_labels))
    if num_classes > max_classes:
        print(f"Too many classes ({num_classes}) for confusion matrix, "
              f"showing top-{max_classes} most frequent classes.")
        # Get top-N most frequent classes
        unique, counts = np.unique(all_labels, return_counts=True)
        top_classes = unique[np.argsort(-counts)[:max_classes]]
        mask = np.isin(all_labels, top_classes)
        all_labels = all_labels[mask]
        all_preds = all_preds[mask]
        used_names = [class_names[i] for i in sorted(top_classes)]
    else:
        used_names = class_names

    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(max(10, len(used_names) * 0.5),
                                     max(8, len(used_names) * 0.4)))
    sns.heatmap(cm, annot=(len(used_names) <= 20), fmt='d',
                cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_score_distribution(verification_results, save_dir):
    """Plot genuine vs impostor score distributions."""
    genuine = np.array(verification_results['genuine_scores'])
    impostor = np.array(verification_results['impostor_scores'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Score distributions
    axes[0].hist(genuine, bins=50, alpha=0.7, label='Genuine', color='#4CAF50', density=True)
    axes[0].hist(impostor, bins=50, alpha=0.7, label='Impostor', color='#F44336', density=True)
    axes[0].set_xlabel('Cosine Similarity')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Score Distribution: Genuine vs Impostor')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ROC curve
    tar = np.array(verification_results['tar'])
    far = np.array(verification_results['far'])
    axes[1].plot(far, tar, color='#2196F3', linewidth=2)
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[1].set_xlabel('False Accept Rate (FAR)')
    axes[1].set_ylabel('True Accept Rate (TAR)')
    axes[1].set_title(f"ROC Curve (EER={verification_results['eer']:.4f})")
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "verification_analysis.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Verification analysis saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Face Recognition Model")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to original dataset for evaluation.")
    parser.add_argument("--anon_dir", type=str, default=None,
                        help="Directory containing anonymised images.")
    parser.add_argument("--label_file", type=str, default=None,
                        help="CSV file mapping anonymised images to identities.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint.")
    args = parser.parse_args()

    print("=" * 60)
    print("COMP4026 - Student A: Face Recognition Evaluation")
    print("=" * 60)

    # Load model
    model, class_names = load_model(args.checkpoint)
    num_classes = len(class_names)

    all_results = {}

    # ============================================================
    # 1. Evaluate on ORIGINAL test set
    # ============================================================
    if args.data_path:
        print("\n--- Evaluation on Original Test Set ---")
        _, _, test_loader, _ = get_dataloaders(args.data_path)

        # Identification accuracy
        id_results = evaluate_identification(
            model, test_loader, num_classes, desc="Original Test Set"
        )
        print(f"\n[Original Images] Identification Results:")
        print(f"  Top-1 Accuracy: {id_results['accuracy']:.2f}%")
        for k, v in id_results['topk_accuracy'].items():
            print(f"  {k}: {v:.2f}%")
        print(f"  Total samples: {id_results['num_samples']}")

        all_results['original_identification'] = {
            'accuracy': id_results['accuracy'],
            'topk_accuracy': id_results['topk_accuracy'],
            'num_samples': id_results['num_samples'],
        }

        # Verification metrics
        print("\nComputing verification metrics...")
        embeddings, labels = evaluate_verification(
            model, test_loader, desc="Extracting embeddings"
        )
        ver_results = compute_verification_metrics(embeddings, labels)
        print(f"  EER: {ver_results['eer']:.4f}")
        for k, v in ver_results['tar_at_far'].items():
            print(f"  {k}: {v:.4f}")

        all_results['original_verification'] = {
            'eer': ver_results['eer'],
            'tar_at_far': ver_results['tar_at_far'],
        }

        # Plot results
        plot_confusion_matrix(
            id_results['all_labels'], id_results['all_preds'],
            class_names, config.RESULTS_DIR,
            title="Face Recognition - Original Test Set"
        )
        plot_score_distribution(ver_results, config.RESULTS_DIR)

        # Plot training history if available
        history_path = os.path.join(config.RESULTS_DIR, "training_history.json")
        if os.path.exists(history_path):
            plot_training_history(history_path, config.RESULTS_DIR)

    # ============================================================
    # 2. Evaluate on ANONYMISED images
    # ============================================================
    if args.anon_dir:
        print("\n--- Evaluation on Anonymised Images ---")
        anon_results = evaluate_anonymised(
            model, args.anon_dir, args.label_file, class_names
        )

        print(f"\n[Anonymised Images] Results:")
        if anon_results['anonymised_accuracy'] is not None:
            print(f"  Recognition Accuracy: {anon_results['anonymised_accuracy']:.2f}%")
            print(f"  Privacy Protection Rate: {anon_results['privacy_protection_rate']:.2f}%")
            print(f"  Correctly Identified: {anon_results['correctly_identified']}/{anon_results['total_images']}")
            print(f"  Random Chance: {100/num_classes:.2f}%")
        else:
            print("  No labelled anonymised images found for accuracy computation.")

        print(f"\n  Note: {anon_results['note']}")
        all_results['anonymised'] = anon_results

    # ============================================================
    # Save all results
    # ============================================================
    results_path = os.path.join(config.RESULTS_DIR, "evaluation_results.json")
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    clean_results = json.loads(json.dumps(all_results, default=convert))
    with open(results_path, "w") as f:
        json.dump(clean_results, f, indent=2)
    print(f"\nAll results saved to {results_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    if 'original_identification' in all_results:
        print(f"Original Test Accuracy: {all_results['original_identification']['accuracy']:.2f}%")
    if 'anonymised' in all_results and all_results['anonymised']['anonymised_accuracy'] is not None:
        print(f"Anonymised Accuracy:    {all_results['anonymised']['anonymised_accuracy']:.2f}%")
        print(f"Privacy Protection:     {all_results['anonymised']['privacy_protection_rate']:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
