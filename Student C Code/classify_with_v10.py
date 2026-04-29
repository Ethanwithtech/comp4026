import os
import shutil
import random
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

warnings.filterwarnings("ignore")


class CFG:
    seed = 42

    # input_dir = "./test/pins_for_anon/images"
    input_dir = "./test/anonymized"
    output_dir = "./results_v10"
    model_weight = "./results_v10/best_model_v10.pt"

    img_size = 96
    dropout = 0.3
    num_classes = 7
    pretrained = False

    classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    batch_size = 128
    num_workers = 4
    pin_memory = True
    preload_to_memory = True


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(CFG.seed)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True


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


class FERModel(nn.Module):
    """SE-ResNet34 classifier used by v10.py."""

    def __init__(self):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if CFG.pretrained else None
        base = models.resnet34(weights=weights)

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        self.layer1 = base.layer1
        self.se1 = SEBlock(64)
        self.layer2 = base.layer2
        self.se2 = SEBlock(128)
        self.layer3 = base.layer3
        self.se3 = SEBlock(256)
        self.layer4 = base.layer4
        self.se4 = SEBlock(512)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(CFG.dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(CFG.dropout * 0.5),
            nn.Linear(256, CFG.num_classes),
        )

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.se1(self.layer1(x))
        x = self.se2(self.layer2(x))
        x = self.se3(self.layer3(x))
        x = self.se4(self.layer4(x))
        x = self.avgpool(x).flatten(1)
        x = self.classifier(x)
        return x


def get_val_transform():
    return T.Compose([
        T.Resize((CFG.img_size, CFG.img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def is_image_file(path: Path) -> bool:
    valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    return path.is_file() and path.suffix.lower() in valid_suffixes


def load_model():
    if not os.path.exists(CFG.model_weight):
        raise FileNotFoundError(f"Cannot find model weights: {CFG.model_weight}")

    model = FERModel().to(device)
    state_dict = torch.load(CFG.model_weight, map_location=device)

    # Support both plain state_dict and checkpoint dict
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict)
    model.eval()

    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    return model


class MemoryImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.items = []
        self.failed = []

        for p in image_paths:
            try:
                img = Image.open(p).convert("RGB")
                tensor = transform(img)
                self.items.append((p, tensor))
            except (UnidentifiedImageError, OSError) as e:
                self.failed.append((p, str(e)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, tensor = self.items[idx]
        return str(path), tensor


class LazyImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        p = self.image_paths[idx]
        img = Image.open(p).convert("RGB")
        tensor = self.transform(img)
        return str(p), tensor


def prepare_result_folders():
    os.makedirs(CFG.output_dir, exist_ok=True)
    for cls_name in CFG.classes:
        os.makedirs(os.path.join(CFG.output_dir, cls_name), exist_ok=True)


@torch.no_grad()
def batch_predict_and_save(model, loader):
    summary = {cls_name: 0 for cls_name in CFG.classes}
    records = []
    use_amp = device.type == "cuda"

    for batch_paths, batch_images in loader:
        if device.type == "cuda":
            batch_images = batch_images.to(device, non_blocking=True, memory_format=torch.channels_last)
        else:
            batch_images = batch_images.to(device)

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(batch_images)
        else:
            logits = model(batch_images)

        probs = torch.softmax(logits, dim=1)
        pred_idxs = torch.argmax(probs, dim=1).cpu().numpy()
        confs = probs.max(dim=1).values.cpu().numpy()

        for img_path_str, pred_idx, conf in zip(batch_paths, pred_idxs, confs):
            img_path = Path(img_path_str)
            pred_class = CFG.classes[int(pred_idx)]

            target_path = Path(CFG.output_dir) / pred_class / img_path.name
            shutil.copy2(img_path, target_path)

            summary[pred_class] += 1
            records.append((img_path.name, pred_class, float(conf)))

    return summary, records


def classify_folder():
    input_dir = Path(CFG.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {CFG.input_dir}")

    prepare_result_folders()
    transform = get_val_transform()
    model = load_model()

    image_paths = sorted([p for p in input_dir.iterdir() if is_image_file(p)])

    if len(image_paths) == 0:
        print(f"No image files found in {CFG.input_dir}.")
        return

    print(f"Found {len(image_paths)} images.")

    if CFG.preload_to_memory:
        print("Loading and preprocessing all images into memory...")
        dataset = MemoryImageDataset(image_paths, transform)
        for p, err in dataset.failed:
            print(f"[Skip] Failed to read image: {p.name} | {err}")
        print(f"Successfully loaded: {len(dataset)} images")
    else:
        dataset = LazyImageDataset(image_paths, transform)

    loader = DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=0 if CFG.preload_to_memory else CFG.num_workers,
        pin_memory=(CFG.pin_memory and device.type == "cuda"),
        persistent_workers=(False if CFG.preload_to_memory else CFG.num_workers > 0),
    )

    print("Starting batch classification...")
    summary, records = batch_predict_and_save(model, loader)

    print("\nClassification complete. Summary:")
    for cls_name in CFG.classes:
        print(f"{cls_name:>10s}: {summary[cls_name]}")

    record_file = Path(CFG.output_dir) / "prediction_results_anonymized.csv"
    with open(record_file, "w", encoding="utf-8") as f:
        f.write("filename,predicted_class,confidence\n")
        for filename, pred_class, confidence in records:
            f.write(f"{filename},{pred_class},{confidence:.6f}\n")

    print(f"\nPrediction records saved to: {record_file}")


if __name__ == "__main__":
    classify_folder()
