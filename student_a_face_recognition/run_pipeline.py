"""
Complete Pipeline: Download -> Train -> Evaluate
COMP4026 Group Project - Student A: Face Recognition Model

This script runs the entire pipeline end-to-end:
1. Download the Pins Face Recognition dataset
2. Train the face recognition model
3. Evaluate on the original test set
4. Generate visualisations and reports

Usage:
    python run_pipeline.py
    python run_pipeline.py --quick  # Quick test with fewer epochs and identities
"""

import argparse
import json
import os
import sys
import time

import config


def main():
    parser = argparse.ArgumentParser(description="Run complete face recognition pipeline")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test mode (fewer epochs, fewer identities)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to dataset (skip download)")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training (use existing model)")
    args = parser.parse_args()

    print("=" * 70)
    print("  COMP4026 - Student A: Face Recognition Pipeline")
    print("  Anonymised Facial Expression Recognition Project")
    print("=" * 70)
    print()

    start_time = time.time()

    # ---- Quick test mode ----
    if args.quick:
        print("[Quick Test Mode] Using reduced settings for fast testing")
        config.NUM_IDENTITIES = 20
        config.NUM_EPOCHS = 5
        config.BATCH_SIZE = 16
        config.MIN_IMAGES_PER_IDENTITY = 3
        print(f"  Identities: {config.NUM_IDENTITIES}")
        print(f"  Epochs: {config.NUM_EPOCHS}")
        print(f"  Batch size: {config.BATCH_SIZE}")
        print()

    # ============================================================
    # Step 1: Download Dataset
    # ============================================================
    print("=" * 50)
    print("Step 1: Preparing Dataset")
    print("=" * 50)

    if args.data_path:
        data_path = args.data_path
        print(f"Using provided dataset path: {data_path}")
    else:
        from utils.dataset import download_dataset
        print("Downloading Pins Face Recognition dataset from Kaggle...")
        data_path = download_dataset()

    # Explore dataset
    from utils.dataset import discover_dataset
    identity_images, class_names = discover_dataset(data_path)
    print(f"Number of identities: {len(class_names)}")
    total_imgs = sum(len(v) for v in identity_images.values())
    print(f"Total images: {total_imgs}")

    # Save dataset info
    dataset_info = {
        'data_path': data_path,
        'num_identities': len(class_names),
        'total_images': total_imgs,
        'images_per_identity': {
            name: len(imgs) for name, imgs in identity_images.items()
        },
    }
    info_path = os.path.join(config.RESULTS_DIR, "dataset_info.json")
    with open(info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)
    print(f"Dataset info saved to {info_path}")
    print()

    # ============================================================
    # Step 2: Train Model
    # ============================================================
    if not args.skip_train:
        print("=" * 50)
        print("Step 2: Training Face Recognition Model")
        print("=" * 50)

        # Import and run training
        from train import main as train_main
        sys.argv = ['train.py', '--data_path', data_path]
        if args.quick:
            sys.argv += ['--epochs', str(config.NUM_EPOCHS)]
        train_main()
        print()
    else:
        print("Step 2: Skipping training (using existing model)")
        print()

    # ============================================================
    # Step 3: Evaluate on Original Test Set
    # ============================================================
    print("=" * 50)
    print("Step 3: Evaluating on Original Test Set")
    print("=" * 50)

    from evaluate import main as eval_main
    sys.argv = ['evaluate.py', '--data_path', data_path]
    eval_main()
    print()

    # ============================================================
    # Step 4: Generate Dataset Visualisations
    # ============================================================
    print("=" * 50)
    print("Step 4: Generating Visualisations")
    print("=" * 50)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        # Distribution of images per identity
        counts = [len(imgs) for imgs in identity_images.values()]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(counts, bins=30, color='#2196F3', alpha=0.8, edgecolor='white')
        axes[0].set_xlabel('Number of Images')
        axes[0].set_ylabel('Number of Identities')
        axes[0].set_title('Distribution of Images per Identity')
        axes[0].axvline(np.mean(counts), color='red', linestyle='--',
                        label=f'Mean: {np.mean(counts):.1f}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Box plot
        axes[1].boxplot(counts, vert=True)
        axes[1].set_ylabel('Number of Images')
        axes[1].set_title('Images per Identity (Box Plot)')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        vis_path = os.path.join(config.RESULTS_DIR, "dataset_distribution.png")
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Dataset distribution plot saved to {vis_path}")

        # Sample images visualisation
        from PIL import Image
        from utils.dataset import get_transforms
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        sample_names = class_names[:10]
        for i, name in enumerate(sample_names):
            row, col = i // 5, i % 5
            img_path = identity_images[name][0]
            img = Image.open(img_path).convert("RGB")
            axes[row, col].imshow(img)
            short_name = name.replace('pins_', '').replace('_', ' ')[:15]
            axes[row, col].set_title(short_name, fontsize=8)
            axes[row, col].axis('off')

        plt.suptitle('Sample Images from Dataset', fontsize=14)
        plt.tight_layout()
        sample_path = os.path.join(config.RESULTS_DIR, "sample_images.png")
        plt.savefig(sample_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Sample images saved to {sample_path}")
    except Exception as e:
        print(f"Warning: Could not generate visualisations: {e}")

    # ============================================================
    # Summary
    # ============================================================
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Results directory: {config.RESULTS_DIR}")
    print(f"  Model directory: {config.MODEL_DIR}")
    print()
    print("  Generated files:")
    for d in [config.RESULTS_DIR, config.MODEL_DIR]:
        if os.path.exists(d):
            for f in sorted(os.listdir(d)):
                print(f"    - {os.path.join(d, f)}")
    print()
    print("  To evaluate on anonymised images (from Student B):")
    print("    python evaluate.py --anon_dir <path> --label_file <csv>")
    print("=" * 70)


if __name__ == "__main__":
    main()
