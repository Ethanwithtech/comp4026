import os, sys, time, random, warnings, contextlib
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


class CFG:
    seed          = 42
    data_dir      = "fer2013"
    output_dir    = "./results_v10"

    img_size      = 96
    num_classes   = 7
    pretrained    = True
    dropout       = 0.3

    batch_size    = 64
    num_workers   = 0
    pin_memory    = False

    stage1_epochs = 5
    stage1_lr     = 1e-3

    stage2_epochs = 80
    stage2_lr     = 8e-4
    backbone_lr_mult = 0.2
    warmup_epochs = 5
    weight_decay  = 1e-4

    label_smooth  = 0.05
    mixup_prob    = 0.2
    mixup_alpha   = 0.3
    cutmix_prob   = 0.2
    cutmix_alpha  = 1.0

    patience      = 25
    tta_augments  = 10

    classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


seed_everything(CFG.seed)
os.makedirs(CFG.output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")

use_amp    = False
use_scaler = False
amp_ctx    = contextlib.nullcontext  

if device.type == "cuda":
    try:
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            _ = torch.ones(1, device=device) + torch.ones(1, device=device)
        use_amp    = True
        use_scaler = True
        amp_ctx    = lambda: torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        print("[AMP] torch.amp.autocast(cuda) + GradScaler ✓")
    except Exception:
        try:
            from torch.cuda.amp import autocast as cuda_autocast
            use_amp    = True
            use_scaler = True
            amp_ctx    = cuda_autocast
            print("[AMP] torch.cuda.amp.autocast + GradScaler ✓ (legacy)")
        except Exception:
            print("[AMP] Not available on CUDA, using float32")

elif device.type == "mps":
    try:
        with torch.amp.autocast(device_type="mps", dtype=torch.float16):
            _ = torch.ones(1, device=device) + torch.ones(1, device=device)
        use_amp = True
        amp_ctx = lambda: torch.amp.autocast(device_type="mps", dtype=torch.float16)
        print("[AMP] torch.amp.autocast(mps) ✓")
    except Exception:
        print("[AMP] MPS autocast not supported, using float32 (all compute on MPS)")

print(f"Device : {device}")
print(f"AMP    : {use_amp}  |  GradScaler: {use_scaler}")
if device.type == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name()}")
elif device.type == "mps":
    print("GPU    : Apple Silicon (MPS)")

class FERDataset(Dataset):
    def __init__(self, pixels_list, labels, transform=None):
        self.pixels = pixels_list
        self.labels = labels
        self.transform = transform
        self.images = []
        for p in pixels_list:
            arr = np.fromstring(p, sep=" ", dtype=np.uint8).reshape(48, 48)
            img = Image.fromarray(arr, mode="L").convert("RGB")
            if CFG.img_size != 48:
                img = img.resize((CFG.img_size, CFG.img_size), Image.BILINEAR)
            self.images.append(img)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


class PreloadedDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        lbl = self.labels[idx]
        if not isinstance(lbl, torch.Tensor):
            lbl = torch.tensor(lbl, dtype=torch.long)
        return img, lbl


def load_data():
    csv_path = os.path.join(CFG.data_dir, "fer2013.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV: {len(df)} samples")
        train_df = df[df["Usage"] == "Training"]
        val_df   = df[df["Usage"] == "PublicTest"]
        test_df  = df[df["Usage"] == "PrivateTest"]
        return (
            (train_df["pixels"].tolist(), train_df["emotion"].tolist()),
            (val_df["pixels"].tolist(),   val_df["emotion"].tolist()),
            (test_df["pixels"].tolist(),  test_df["emotion"].tolist()),
        )
    print("CSV not found, trying image folders...")
    train_data, test_data = [], []
    for split_name, container in [("train", train_data), ("test", test_data)]:
        split_dir = os.path.join(CFG.data_dir, split_name)
        if not os.path.exists(split_dir):
            continue
        px, lb = [], []
        for ci, cn in enumerate(CFG.classes):
            cls_dir = os.path.join(split_dir, cn)
            if not os.path.exists(cls_dir):
                continue
            for fn in sorted(os.listdir(cls_dir)):
                img = Image.open(os.path.join(cls_dir, fn)).convert("L")
                px.append(" ".join(map(str, np.array(img).flatten())))
                lb.append(ci)
        container.extend([px, lb])
    if train_data and test_data:
        return (
            (train_data[0], train_data[1]),
            (test_data[0],  test_data[1]),
            (test_data[0],  test_data[1]),
        )
    raise FileNotFoundError("No data found!")

def get_train_transform():
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.ColorJitter(brightness=0.25, contrast=0.25),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])


def get_val_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def get_tta_transform():
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(10),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def _one_hot(labels, num_classes):
    return torch.zeros(labels.size(0), num_classes,
                       device=labels.device, dtype=torch.float32) \
               .scatter_(1, labels.unsqueeze(1), 1.0)


def _rand_bbox(H, W, lam):
    cut_rat = (1.0 - lam) ** 0.5
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = random.randint(0, W - 1)
    cy = random.randint(0, H - 1)
    x1 = max(0, cx - cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    x2 = min(W, cx + cut_w // 2)
    y2 = min(H, cy + cut_h // 2)
    return x1, y1, x2, y2


def apply_mixup_cutmix(images, labels, num_classes):
    r = random.random()
    if r < CFG.cutmix_prob:
        lam = random.betavariate(CFG.cutmix_alpha, CFG.cutmix_alpha)
        idx = torch.randperm(images.size(0), device=images.device)
        H, W = images.size(2), images.size(3)
        x1, y1, x2, y2 = _rand_bbox(H, W, lam)
        images[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
        lam = 1.0 - float((x2 - x1) * (y2 - y1)) / float(H * W)
        tgt = lam * _one_hot(labels, num_classes) + \
              (1.0 - lam) * _one_hot(labels[idx], num_classes)
        return images, tgt
    elif r < CFG.cutmix_prob + CFG.mixup_prob:
        lam = random.betavariate(CFG.mixup_alpha, CFG.mixup_alpha)
        idx = torch.randperm(images.size(0), device=images.device)
        images = lam * images + (1.0 - lam) * images[idx]
        tgt = lam * _one_hot(labels, num_classes) + \
              (1.0 - lam) * _one_hot(labels[idx], num_classes)
        return images, tgt
    else:
        return images, _one_hot(labels, num_classes)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class SEResNet34(nn.Module):
    def __init__(self, num_classes=7, dropout=0.3, pretrained=True):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        base = models.resnet34(weights=weights)

        self.conv1   = base.conv1
        self.bn1     = base.bn1
        self.relu    = base.relu
        self.maxpool = base.maxpool

        self.layer1 = base.layer1;  self.se1 = SEBlock(64)
        self.layer2 = base.layer2;  self.se2 = SEBlock(128)
        self.layer3 = base.layer3;  self.se3 = SEBlock(256)
        self.layer4 = base.layer4;  self.se4 = SEBlock(512)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.se1(self.layer1(x))
        x = self.se2(self.layer2(x))
        x = self.se3(self.layer3(x))
        x = self.se4(self.layer4(x))
        x = self.avgpool(x).flatten(1)
        return self.classifier(x)

    def freeze_backbone(self):
        for name, p in self.named_parameters():
            p.requires_grad = ("se" in name or "classifier" in name)

    def unfreeze_backbone(self):
        for p in self.parameters():
            p.requires_grad = True

    def get_param_groups(self, lr):
        backbone, new = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "se" in name or "classifier" in name:
                new.append(p)
            else:
                backbone.append(p)
        return [
            {"params": backbone, "lr": lr * CFG.backbone_lr_mult},
            {"params": new,      "lr": lr},
        ]


def soft_cross_entropy(logits, targets, label_smoothing=0.0):
    if label_smoothing > 0:
        nc = logits.size(-1)
        targets = targets * (1.0 - label_smoothing) + label_smoothing / nc
    log_probs = F.log_softmax(logits, dim=-1)
    return -(targets * log_probs).sum(-1).mean()


def _make_scaler():
    """Create GradScaler only for CUDA."""
    if not use_scaler:
        return None
    try:
        return torch.amp.GradScaler(device="cuda")
    except TypeError:
        return torch.cuda.amp.GradScaler()


def train_one_epoch(model, loader, optimizer, scheduler, scaler,
                    use_mixup=True):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_mixup:
            images, targets = apply_mixup_cutmix(images, labels, CFG.num_classes)
        else:
            targets = _one_hot(labels, CFG.num_classes)

        optimizer.zero_grad(set_to_none=True)

        with amp_ctx():
            logits = model(images)
            loss = soft_cross_entropy(logits, targets, CFG.label_smooth)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct    += (logits.detach().argmax(1) == labels).sum().item()
        total      += labels.size(0)

    if scheduler is not None:
        scheduler.step()
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with amp_ctx():
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        total_loss += loss.item() * images.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate_tta(model, dataset, n_aug=10):
    model.eval()
    n = len(dataset)
    all_probs  = torch.zeros(n, CFG.num_classes, device=device)
    all_labels = torch.zeros(n, dtype=torch.long, device=device)

    def make_loader(tfm):
        ds = PreloadedDataset(dataset.images, dataset.labels, tfm)
        return DataLoader(ds, CFG.batch_size, shuffle=False,
                          num_workers=CFG.num_workers)

    idx = 0
    for imgs, lbls in make_loader(get_val_transform()):
        imgs = imgs.to(device, non_blocking=True)
        bs = imgs.size(0)
        with amp_ctx():
            probs = F.softmax(model(imgs), dim=-1)
        all_probs[idx:idx+bs]  += probs.float()
        all_labels[idx:idx+bs]  = lbls.to(device, non_blocking=True)
        idx += bs

    for aug_i in range(n_aug):
        idx = 0
        for imgs, _ in make_loader(get_tta_transform()):
            imgs = imgs.to(device, non_blocking=True)
            bs = imgs.size(0)
            with amp_ctx():
                probs = F.softmax(model(imgs), dim=-1)
            all_probs[idx:idx+bs] += probs.float()
            idx += bs

    preds = all_probs.argmax(dim=1)
    acc = 100.0 * (preds == all_labels).float().mean().item()
    return acc, preds.cpu().numpy(), all_labels.cpu().numpy()


def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    sb = CFG.stage1_epochs

    axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   "r-", label="Val Loss")
    axes[0].axvline(sb, color="gray", ls="--", alpha=.5, label="Stage 1→2")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Loss"); axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], "b-", label="Train Acc")
    axes[1].plot(epochs, history["val_acc"],   "r-", label="Val Acc")
    axes[1].axvline(sb, color="gray", ls="--", alpha=.5)
    axes[1].set(xlabel="Epoch", ylabel="Acc (%)", title="Accuracy"); axes[1].legend()

    axes[2].plot(epochs, history["lr"], "g-")
    axes[2].axvline(sb, color="gray", ls="--", alpha=.5)
    axes[2].set(xlabel="Epoch", ylabel="LR", title="Learning Rate")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved → {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(cm,   annot=True, fmt="d",   cmap="Blues",
                xticklabels=CFG.classes, yticklabels=CFG.classes, ax=axes[0])
    axes[0].set(xlabel="Predicted", ylabel="True", title="Confusion Matrix (Counts)")
    sns.heatmap(cm_n, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=CFG.classes, yticklabels=CFG.classes, ax=axes[1])
    axes[1].set(xlabel="Predicted", ylabel="True", title="Confusion Matrix (Normalized)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved → {save_path}")


def plot_per_class_accuracy(y_true, y_pred, overall_acc, save_path):
    cm = confusion_matrix(y_true, y_pred)
    pca = cm.diagonal() / cm.sum(axis=1) * 100
    colors = ["#e74c3c","#9b59b6","#3498db","#2ecc71","#1abc9c","#f39c12","#95a5a6"]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(CFG.classes, pca, color=colors)
    for bar, v in zip(bars, pca):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f"{v:.1f}%", ha="center", fontweight="bold")
    ax.set(ylim=[0, 100], ylabel="Accuracy (%)",
           title=f"Per-Class Accuracy (Overall: {overall_acc:.1f}%)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved → {save_path}")


def main():
    print("=" * 60)
    print("FER v10 — SE-ResNet34 + Balanced Reg + Full GPU")
    print("=" * 60)

    (trn_pix, trn_lbl), (val_pix, val_lbl), (tst_pix, tst_lbl) = load_data()
    print(f"Train: {len(trn_lbl)}  Val: {len(val_lbl)}  Test: {len(tst_lbl)}")
    print(f"Resolution: {CFG.img_size}×{CFG.img_size}")

    print("\nPre-loading images into RAM...")
    train_ds = FERDataset(trn_pix, trn_lbl, get_train_transform())
    val_ds   = FERDataset(val_pix, val_lbl, get_val_transform())
    test_ds  = FERDataset(tst_pix, tst_lbl, get_val_transform())
    print("Done.")

    train_ld = DataLoader(train_ds, CFG.batch_size, shuffle=True,
                          num_workers=CFG.num_workers, pin_memory=CFG.pin_memory,
                          drop_last=True)
    val_ld   = DataLoader(val_ds,  CFG.batch_size, shuffle=False,
                          num_workers=CFG.num_workers)
    test_ld  = DataLoader(test_ds, CFG.batch_size, shuffle=False,
                          num_workers=CFG.num_workers)

    model = SEResNet34(CFG.num_classes, CFG.dropout, CFG.pretrained).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_new   = sum(p.numel() for n, p in model.named_parameters()
                  if "se" in n or "classifier" in n)
    print(f"\nSE-ResNet34: {n_total:,} total params "
          f"(pretrained: {n_total-n_new:,}, new: {n_new:,})")

    scaler = _make_scaler()
    hist = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[], "lr":[]}
    best_va, no_imp = 0.0, 0
    ckpt_path = os.path.join(CFG.output_dir, "best_model_v10.pt")
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"STAGE 1: SE + Classifier head only ({CFG.stage1_epochs} epochs)")
    print(f"{'='*60}")

    model.freeze_backbone()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {n_train:,} / {n_total:,}")

    opt1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG.stage1_lr, weight_decay=CFG.weight_decay)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=CFG.stage1_epochs)

    for ep in range(1, CFG.stage1_epochs + 1):
        tl, ta = train_one_epoch(model, train_ld, opt1, sch1, scaler,
                                 use_mixup=False)
        vl, va = evaluate(model, val_ld)
        lr_now = opt1.param_groups[0]["lr"]
        hist["train_loss"].append(tl); hist["val_loss"].append(vl)
        hist["train_acc"].append(ta);  hist["val_acc"].append(va)
        hist["lr"].append(lr_now)

        tag = ""
        if va > best_va:
            best_va = va
            torch.save(model.state_dict(), ckpt_path)
            tag = " ★"
        print(f"  [{ep}/{CFG.stage1_epochs}] "
              f"Train {tl:.4f}/{ta:.1f}%  Val {vl:.4f}/{va:.1f}%  "
              f"LR {lr_now:.6f}  Best {best_va:.1f}%{tag}")

    print(f"\n{'='*60}")
    print(f"STAGE 2: Full fine-tune ({CFG.stage2_epochs} epochs, "
          f"warmup={CFG.warmup_epochs}, patience={CFG.patience})")
    print(f"{'='*60}")

    model.unfreeze_backbone()
    pg = model.get_param_groups(CFG.stage2_lr)
    print(f"  Backbone LR: {pg[0]['lr']:.6f}  |  New LR: {pg[1]['lr']:.6f}")

    opt2 = torch.optim.AdamW(pg, weight_decay=CFG.weight_decay)

    try:
        sch_warmup = torch.optim.lr_scheduler.LinearLR(
            opt2, start_factor=0.1, total_iters=CFG.warmup_epochs)
        sch_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt2, T_max=CFG.stage2_epochs - CFG.warmup_epochs, eta_min=1e-6)
        sch2 = torch.optim.lr_scheduler.SequentialLR(
            opt2, schedulers=[sch_warmup, sch_cosine],
            milestones=[CFG.warmup_epochs])
        print(f"  Schedule: Warmup({CFG.warmup_epochs}ep) → CosineAnnealing")
    except AttributeError:
        sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt2, T_max=CFG.stage2_epochs, eta_min=1e-6)
        print(f"  Schedule: CosineAnnealing (fallback)")

    for ep in range(1, CFG.stage2_epochs + 1):
        gep = CFG.stage1_epochs + ep
        tl, ta = train_one_epoch(model, train_ld, opt2, sch2, scaler,
                                 use_mixup=True)
        vl, va = evaluate(model, val_ld)
        lr_now = opt2.param_groups[0]["lr"]
        hist["train_loss"].append(tl); hist["val_loss"].append(vl)
        hist["train_acc"].append(ta);  hist["val_acc"].append(va)
        hist["lr"].append(lr_now)

        tag = ""
        if va > best_va:
            best_va = va
            torch.save(model.state_dict(), ckpt_path)
            tag = " ★"
            no_imp = 0
        else:
            no_imp += 1

        if ep % 5 == 0 or ep <= 3 or tag or no_imp >= CFG.patience:
            mins = (time.time() - t0) / 60
            print(f"  [{gep:2d}/{CFG.stage1_epochs+CFG.stage2_epochs}] "
                  f"Train {tl:.4f}/{ta:.1f}%  Val {vl:.4f}/{va:.1f}%  "
                  f"LR {lr_now:.6f}  Best {best_va:.1f}%  "
                  f"Pat {no_imp}/{CFG.patience}  {mins:.1f}min{tag}")

        if no_imp >= CFG.patience:
            print(f"\n  ⏹ Early stopping at epoch {gep}")
            break

    elapsed = (time.time() - t0) / 60
    print(f"\nTraining done: {elapsed:.1f} min | Best Val: {best_va:.2f}%")
    plot_training_curves(hist,
                         os.path.join(CFG.output_dir, "v10_training_curves.png"))

    print(f"\n{'='*60}")
    print("FINAL TEST EVALUATION")
    print(f"{'='*60}")

    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True))

    _, test_acc = evaluate(model, test_ld)
    print(f"  Standard: {test_acc:.2f}%")

    print(f"  TTA ({CFG.tta_augments} augmentations)...")
    tta_acc, y_pred, y_true = evaluate_tta(model, test_ds, CFG.tta_augments)
    print(f"  TTA:      {tta_acc:.2f}%")

    print(f"\n{classification_report(y_true, y_pred, target_names=CFG.classes, digits=4)}")

    plot_confusion_matrix(y_true, y_pred,
                          os.path.join(CFG.output_dir, "v10_confusion_matrix.png"))
    plot_per_class_accuracy(y_true, y_pred, tta_acc,
                            os.path.join(CFG.output_dir, "v10_per_class_acc.png"))

    print(f"\n{'='*60}")
    print("VERSION HISTORY")
    print(f"{'='*60}")
    print(f"  v4  (ResNet18):          71.47% TTA")
    print(f"  v8  (SE-ResNet18):       71.52% TTA")
    print(f"  v9  (EfficientNet-B0):   70.21% TTA  ← over-regularized")
    print(f"  v10 (SE-ResNet34):       {tta_acc:.2f}% TTA  (std: {test_acc:.2f}%)")
    print(f"  v4→v10:                  {tta_acc - 71.47:+.2f}%")
    print(f"{'='*60}")
    print("Done!")


if __name__ == "__main__":
    main()