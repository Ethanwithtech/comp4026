# Student A: Face Recognition Model

## COMP4026 Group Project — Anonymised Facial Expression Recognition

### Overview

This module implements a **face recognition model** to verify the identity protection strength of the anonymisation pipeline. The model serves two purposes:

1. **Validate the ID classifier reliability**: Achieve high accuracy on original face images to confirm the model can reliably recognise identities.
2. **Evaluate anonymisation strength**: Test the trained model on anonymised images — a lower recognition accuracy indicates stronger privacy protection.

### Architecture

- **Backbone**: ResNet-50 (pretrained on ImageNet)
- **Embedding**: 512-dimensional L2-normalised face embedding
- **Loss**: ArcFace (Additive Angular Margin Loss) for discriminative training
- **Dataset**: [Pins Face Recognition](https://www.kaggle.com/datasets/hereisburak/pins-face-recognition) from Kaggle

```
Input (3 × 160 × 160)
  → ResNet-50 backbone
  → Global Average Pooling
  → FC (2048 → 512) + BatchNorm
  → L2-Normalised Embedding (512-d)
  → ArcFace Classification Head
```

### Project Structure

```
student_a_face_recognition/
├── config.py            # Hyperparameters, paths, device settings
├── model.py             # FaceRecognitionModel + ArcFace loss
├── train.py             # Training script
├── evaluate.py          # Evaluation on original & anonymised images
├── inference.py         # Easy-to-use inference API (FaceRecognizer class)
├── run_pipeline.py      # One-click: download → train → evaluate
├── requirements.txt     # Python dependencies
├── utils/
│   ├── __init__.py
│   └── dataset.py       # Dataset loading, splitting, augmentation
├── data/                # (auto-created) Downloaded dataset
├── models/              # (auto-created) Saved model checkpoints
└── results/             # (auto-created) Evaluation results & plots
```

### Quick Start

#### 1. Install Dependencies

```bash
cd student_a_face_recognition
pip install -r requirements.txt
```

#### 2. Run Full Pipeline (Recommended)

```bash
# Full training (30 epochs)
python run_pipeline.py

# Quick test mode (5 epochs, 20 identities) — for rapid prototyping
python run_pipeline.py --quick
```

#### 3. Step-by-Step

```bash
# Train only
python train.py --data_path /path/to/pins/dataset

# Evaluate on original test set
python evaluate.py --data_path /path/to/pins/dataset

# Evaluate anonymised images (after Student B provides them)
python evaluate.py --anon_dir /path/to/anonymised/ --label_file labels.csv
```

### API for Team Members (Student B & C)

The `FaceRecognizer` class in `inference.py` provides a clean API:

```python
from inference import FaceRecognizer

# Initialise
recognizer = FaceRecognizer()

# 1. Predict identity
result = recognizer.predict("face.jpg")
print(result['identity'], result['confidence'])

# 2. Extract embedding (512-d vector)
embedding = recognizer.get_embedding("face.jpg")

# 3. Verify if two faces are the same person
result = recognizer.verify("face1.jpg", "face2.jpg")
print(result['same_person'], result['similarity'])

# 4. Evaluate anonymisation quality
results = recognizer.evaluate_anonymisation(
    original_dir="original_faces/",
    anonymised_dir="anonymised_faces/"
)
print(f"Privacy Protection Rate: {results['privacy_protection_rate']:.1f}%")
```

### Evaluation Metrics

#### Face Recognition Metrics (Privacy)

| Metric | Description | Expected |
|--------|-------------|----------|
| **Accuracy (original)** | ID accuracy on original test images | High (>90%) |
| **Accuracy (anonymised)** | ID accuracy on anonymised images | Low (near random) |
| **Privacy Protection Rate** | 100% - anonymised accuracy | High (>90%) |
| **EER** | Equal Error Rate for verification | Low |
| **TAR@FAR** | True Accept Rate at fixed False Accept Rate | High on originals |

#### Interpretation

- **High accuracy on originals** → The model is a reliable identity classifier.
- **Low accuracy on anonymised images** → The anonymisation successfully hides identity.
- The gap between original and anonymised accuracy quantifies anonymisation strength.

### Configuration

All hyperparameters are in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BACKBONE` | resnet50 | Feature extractor |
| `EMBEDDING_DIM` | 512 | Face embedding dimension |
| `LOSS_TYPE` | arcface | ArcFace angular margin loss |
| `BATCH_SIZE` | 32 | Training batch size |
| `NUM_EPOCHS` | 30 | Training epochs |
| `LEARNING_RATE` | 1e-3 | Initial learning rate |
| `IMAGE_SIZE` | 160 | Input image resolution |
| `ARCFACE_S` | 30.0 | ArcFace scale factor |
| `ARCFACE_M` | 0.5 | ArcFace angular margin |

### Output Files

After training and evaluation:

- `models/best_model.pth` — Best model checkpoint
- `models/class_names.json` — Identity class name mapping
- `results/training_history.json` — Epoch-wise loss/accuracy
- `results/training_results.json` — Final training metrics
- `results/evaluation_results.json` — Complete evaluation report
- `results/training_curves.png` — Loss and accuracy plots
- `results/confusion_matrix.png` — Classification confusion matrix
- `results/verification_analysis.png` — Genuine vs impostor distributions
- `results/dataset_distribution.png` — Images per identity histogram

### GenAI Usage Disclosure

This codebase was developed with the assistance of GenAI tools for code generation, documentation writing, and architectural design. The core methodology (ResNet-50 + ArcFace) follows established academic literature in face recognition.

### References

- Deng, J., et al. "ArcFace: Additive Angular Margin Loss for Deep Face Recognition." CVPR, 2019.
- He, K., et al. "Deep Residual Learning for Image Recognition." CVPR, 2016.
- Schroff, F., et al. "FaceNet: A Unified Embedding for Face Recognition and Clustering." CVPR, 2015.
