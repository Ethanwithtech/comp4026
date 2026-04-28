"""
Face Recognition Model for Student A.
Architecture: ResNet50 backbone + ArcFace classification head.
COMP4026 Group Project: Anonymised Facial Expression Recognition
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import config


# ============================================================
# ArcFace Loss (Additive Angular Margin Loss)
# ============================================================
class ArcFaceHead(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss for Deep Face Recognition.
    Reference: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep
    Face Recognition", CVPR 2019.

    This loss enforces a margin in the angular space, making the learned
    embeddings more discriminative for face recognition.
    """

    def __init__(self, in_features: int, num_classes: int,
                 s: float = 30.0, m: float = 0.5):
        """
        Args:
            in_features: Embedding dimension
            num_classes: Number of identities
            s: Scale factor
            m: Angular margin (in radians)
        """
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m

        # Weight matrix for the classifier
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Precompute cos/sin of margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, in_features) normalised face embeddings
            labels: (B,) ground truth identity labels
        Returns:
            logits: (B, num_classes) scaled cosine logits with angular margin
        """
        # Normalise embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity
        cosine = F.linear(embeddings, weight)  # (B, num_classes)
        sine = torch.sqrt(1.0 - torch.clamp(cosine * cosine, 0, 1))

        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # For numerical stability
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot encoding
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Apply margin only to the target class
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.s

        return logits


# ============================================================
# Face Recognition Model
# ============================================================
class FaceRecognitionModel(nn.Module):
    """
    Face Recognition Model using ResNet50 backbone.

    Architecture:
        Input (3 x 160 x 160)
        -> ResNet50 (pretrained on ImageNet)
        -> Global Average Pooling
        -> Fully Connected (2048 -> 512)
        -> BatchNorm
        -> Embedding (512-d, L2-normalised)

    During training:
        Embedding -> ArcFace Head -> Cross-Entropy Loss

    During inference:
        Embedding -> Cosine Similarity for verification/identification
    """

    def __init__(self, num_classes: int, embedding_dim: int = 512,
                 backbone: str = "resnet50", pretrained: bool = True):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        # ---- Backbone ----
        if backbone == "resnet50":
            base_model = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            )
            backbone_out_features = base_model.fc.in_features  # 2048
            # Remove the original FC layer
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        elif backbone == "mobilenet_v2":
            base_model = models.mobilenet_v2(
                weights=models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
            )
            backbone_out_features = base_model.classifier[1].in_features  # 1280
            self.backbone = nn.Sequential(
                base_model.features,
                nn.AdaptiveAvgPool2d((1, 1)),
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # ---- Embedding Head ----
        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_out_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        # ---- Classification Head ----
        if config.LOSS_TYPE == "arcface":
            self.classifier = ArcFaceHead(
                in_features=embedding_dim,
                num_classes=num_classes,
                s=config.ARCFACE_S,
                m=config.ARCFACE_M,
            )
        else:
            self.classifier = nn.Linear(embedding_dim, num_classes)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract face embedding (for inference/evaluation).
        Args:
            x: (B, 3, H, W) input face images
        Returns:
            embeddings: (B, embedding_dim) L2-normalised embeddings
        """
        features = self.backbone(x)
        embeddings = self.embedding_head(features)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def forward(self, x: torch.Tensor,
                labels: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: (B, 3, H, W) input face images
            labels: (B,) identity labels (required for ArcFace training)
        Returns:
            If training with ArcFace: logits (B, num_classes)
            If eval or CE loss: logits (B, num_classes)
        """
        features = self.backbone(x)
        embeddings = self.embedding_head(features)

        if config.LOSS_TYPE == "arcface" and labels is not None:
            logits = self.classifier(embeddings, labels)
        elif config.LOSS_TYPE == "arcface" and labels is None:
            # During inference, compute cosine similarity directly
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            weight_norm = F.normalize(self.classifier.weight, p=2, dim=1)
            logits = F.linear(embeddings_norm, weight_norm) * self.classifier.s
        else:
            logits = self.classifier(embeddings)

        return logits


def build_model(num_classes: int) -> FaceRecognitionModel:
    """Build and return the face recognition model."""
    model = FaceRecognitionModel(
        num_classes=num_classes,
        embedding_dim=config.EMBEDDING_DIM,
        backbone=config.BACKBONE,
        pretrained=config.PRETRAINED,
    )
    model = model.to(config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {config.BACKBONE} + {'ArcFace' if config.LOSS_TYPE == 'arcface' else 'CE'}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Device: {config.DEVICE}")

    return model
