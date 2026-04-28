"""
Training script for Face Recognition Model (Student A).
COMP4026 Group Project: Anonymised Facial Expression Recognition

Key improvement: Two-phase training
  Phase 1 (warmup): Train with Cross-Entropy loss to get a good initial embedding
  Phase 2 (arcface): Switch to ArcFace loss for discriminative fine-tuning

Usage:
    python train.py [--data_path PATH_TO_DATASET]
"""

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, OneCycleLR
from tqdm import tqdm

import config
from model import build_model
from utils.dataset import download_dataset, discover_dataset, get_dataloaders


def compute_accuracy_no_margin(model, images):
    """
    Compute accuracy using cosine similarity WITHOUT ArcFace margin.
    This gives a true accuracy reading even during ArcFace training.
    """
    model.eval()
    with torch.no_grad():
        embeddings = model.get_embedding(images)
        weight_norm = F.normalize(model.classifier.weight, p=2, dim=1)
        logits = F.linear(embeddings, weight_norm)
    model.train()
    return logits


def train_one_epoch(model, train_loader, criterion, optimizer, epoch,
                    use_arcface=True):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    phase = "ArcFace" if use_arcface else "CE-Warmup"
    pbar = tqdm(train_loader,
                desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [{phase}]")

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        optimizer.zero_grad()

        if use_arcface and config.LOSS_TYPE == "arcface":
            # ArcFace forward (with margin)
            logits = model(images, labels)
            loss = criterion(logits, labels)

            # For accuracy display, compute WITHOUT margin
            clean_logits = compute_accuracy_no_margin(model, images)
            _, predicted = clean_logits.max(1)
        else:
            # Cross-Entropy forward (no margin) — warmup phase
            logits = model(images)  # labels=None -> no margin applied
            loss = criterion(logits, labels)
            _, predicted = logits.max(1)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{running_loss/(batch_idx+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, val_loader, criterion):
    """Validate the model (always without ArcFace margin)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(val_loader, desc="Validating")
    for images, labels in pbar:
        images = images.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        # Always evaluate without margin
        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / len(val_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train Face Recognition Model")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to the Pins Face Recognition dataset. "
                             "If not provided, will download from Kaggle.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs.")
    args = parser.parse_args()

    if args.epochs:
        config.NUM_EPOCHS = args.epochs

    warmup_epochs = getattr(config, 'WARMUP_EPOCHS', 5)

    print("=" * 60)
    print("COMP4026 - Student A: Face Recognition Model Training")
    print("=" * 60)
    print(f"Device: {config.DEVICE}")
    print(f"Backbone: {config.BACKBONE}")
    print(f"Loss: {config.LOSS_TYPE} (CE warmup for first {warmup_epochs} epochs)")
    print(f"ArcFace margin: {config.ARCFACE_M}, scale: {config.ARCFACE_S}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print()

    # ---- Data ----
    if args.data_path:
        data_path = args.data_path
    else:
        print("Downloading dataset from Kaggle...")
        data_path = download_dataset()

    print(f"Dataset path: {data_path}")
    train_loader, val_loader, test_loader, class_names = get_dataloaders(data_path)
    num_classes = len(class_names)
    print(f"Number of identity classes: {num_classes}")
    print()

    # Save class names
    class_names_path = os.path.join(config.MODEL_DIR, "class_names.json")
    with open(class_names_path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"Class names saved to {class_names_path}")

    # ---- Model ----
    model = build_model(num_classes)

    # ---- Loss & Optimizer ----
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Use different LR for backbone (pretrained) vs head (new)
    backbone_params = list(model.backbone.parameters())
    head_params = (list(model.embedding_head.parameters()) +
                   list(model.classifier.parameters()))

    optimizer = AdamW([
        {'params': backbone_params, 'lr': config.LEARNING_RATE * 0.1},
        {'params': head_params, 'lr': config.LEARNING_RATE},
    ], weight_decay=config.WEIGHT_DECAY)

    # ---- Scheduler ----
    if config.LR_SCHEDULER == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6
        )
    else:
        scheduler = StepLR(
            optimizer, step_size=config.LR_STEP_SIZE, gamma=config.LR_GAMMA
        )

    # ---- Resume ----
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"Resumed at epoch {start_epoch}, best val acc: {best_val_acc:.2f}%")

    # ---- Training Loop ----
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': [], 'phase': [],
    }

    print("\nStarting training...")
    print(f"Phase 1 (Epoch 1-{warmup_epochs}): Cross-Entropy warmup")
    print(f"Phase 2 (Epoch {warmup_epochs+1}-{config.NUM_EPOCHS}): ArcFace fine-tuning")
    start_time = time.time()

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        # Determine training phase
        use_arcface = (epoch >= warmup_epochs) and (config.LOSS_TYPE == "arcface")
        phase_name = "arcface" if use_arcface else "ce_warmup"

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch,
            use_arcface=use_arcface,
        )

        # Validate (always without margin)
        val_loss, val_acc = validate(model, val_loader, criterion)

        # Update LR
        current_lr = optimizer.param_groups[1]['lr']  # head LR
        scheduler.step()

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        history['phase'].append(phase_name)

        phase_str = "ArcFace" if use_arcface else "CE-Warmup"
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS} [{phase_str}]:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  LR (head): {current_lr:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(config.MODEL_DIR, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'num_classes': num_classes,
                'class_names': class_names,
                'config': {
                    'backbone': config.BACKBONE,
                    'embedding_dim': config.EMBEDDING_DIM,
                    'loss_type': config.LOSS_TYPE,
                    'image_size': config.IMAGE_SIZE,
                },
            }, best_path)
            print(f"  *** New best model saved (val_acc={best_val_acc:.2f}%) ***")

        # Save latest checkpoint
        latest_path = os.path.join(config.MODEL_DIR, "latest_checkpoint.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'num_classes': num_classes,
            'class_names': class_names,
            'config': {
                'backbone': config.BACKBONE,
                'embedding_dim': config.EMBEDDING_DIM,
                'loss_type': config.LOSS_TYPE,
                'image_size': config.IMAGE_SIZE,
            },
        }, latest_path)

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time/60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

    # Save training history
    history_path = os.path.join(config.RESULTS_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")

    # ---- Final Test Evaluation ----
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)

    best_checkpoint = torch.load(best_path, map_location=config.DEVICE)
    model.load_state_dict(best_checkpoint['model_state_dict'])

    test_loss, test_acc = validate(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")

    # Save final results
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'best_val_accuracy': best_val_acc,
        'num_classes': num_classes,
        'total_training_time_minutes': total_time / 60,
        'config': {
            'backbone': config.BACKBONE,
            'embedding_dim': config.EMBEDDING_DIM,
            'loss_type': config.LOSS_TYPE,
            'batch_size': config.BATCH_SIZE,
            'num_epochs': config.NUM_EPOCHS,
            'learning_rate': config.LEARNING_RATE,
            'arcface_margin': config.ARCFACE_M,
            'arcface_scale': config.ARCFACE_S,
            'warmup_epochs': warmup_epochs,
        },
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    results_path = os.path.join(config.RESULTS_DIR, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
