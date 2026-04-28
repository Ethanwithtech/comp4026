"""
Inference script for Face Recognition Model (Student A).
COMP4026 Group Project: Anonymised Facial Expression Recognition

Provides easy-to-use API for:
1. Identity prediction on single images
2. Batch identity prediction
3. Embedding extraction (for verification)
4. Comparing two faces (verification)

Usage:
    # Predict identity of a single image
    python inference.py --image path/to/face.jpg

    # Compare two faces
    python inference.py --image1 face1.jpg --image2 face2.jpg

    # Batch prediction on a folder
    python inference.py --input_dir path/to/folder --output results.csv
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config
from model import FaceRecognitionModel
from utils.dataset import get_transforms


class FaceRecognizer:
    """
    Face Recognition inference wrapper.
    Provides a clean API for other team members (Student B and C).
    """

    def __init__(self, checkpoint_path: str = None):
        """
        Load the trained face recognition model.

        Args:
            checkpoint_path: Path to model checkpoint. 
                             Defaults to best_model.pth in MODEL_DIR.
        """
        if checkpoint_path is None:
            checkpoint_path = os.path.join(config.MODEL_DIR, "best_model.pth")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                "Please train the model first: python train.py"
            )

        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
        self.class_names = checkpoint['class_names']
        self.num_classes = checkpoint['num_classes']
        model_config = checkpoint.get('config', {})

        self.model = FaceRecognitionModel(
            num_classes=self.num_classes,
            embedding_dim=model_config.get('embedding_dim', config.EMBEDDING_DIM),
            backbone=model_config.get('backbone', config.BACKBONE),
            pretrained=False,
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(config.DEVICE)
        self.model.eval()

        self.transform = get_transforms(is_training=False)
        print(f"FaceRecognizer loaded: {self.num_classes} identities")

    def _preprocess(self, image) -> torch.Tensor:
        """Preprocess a single image."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")

        tensor = self.transform(image).unsqueeze(0)  # (1, 3, H, W)
        return tensor.to(config.DEVICE)

    @torch.no_grad()
    def predict(self, image, top_k: int = 5) -> dict:
        """
        Predict the identity of a face image.

        Args:
            image: PIL Image, numpy array, or file path
            top_k: Number of top predictions to return

        Returns:
            dict with 'identity', 'confidence', 'top_k_predictions'
        """
        tensor = self._preprocess(image)
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1)[0]

        top_probs, top_indices = probs.topk(min(top_k, self.num_classes))

        predictions = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            predictions.append({
                'identity': self.class_names[idx],
                'confidence': float(prob),
                'class_index': int(idx),
            })

        return {
            'identity': predictions[0]['identity'],
            'confidence': predictions[0]['confidence'],
            'top_k_predictions': predictions,
        }

    @torch.no_grad()
    def get_embedding(self, image) -> np.ndarray:
        """
        Extract face embedding for verification.

        Args:
            image: PIL Image, numpy array, or file path

        Returns:
            embedding: (embedding_dim,) L2-normalised numpy array
        """
        tensor = self._preprocess(image)
        embedding = self.model.get_embedding(tensor)
        return embedding[0].cpu().numpy()

    @torch.no_grad()
    def verify(self, image1, image2) -> dict:
        """
        Verify whether two face images belong to the same person.

        Args:
            image1, image2: PIL Image, numpy array, or file path

        Returns:
            dict with 'same_person', 'similarity', 'threshold'
        """
        emb1 = self.get_embedding(image1)
        emb2 = self.get_embedding(image2)

        # Cosine similarity
        similarity = float(np.dot(emb1, emb2))

        return {
            'same_person': similarity >= config.VERIFICATION_THRESHOLD,
            'similarity': similarity,
            'threshold': config.VERIFICATION_THRESHOLD,
        }

    @torch.no_grad()
    def batch_predict(self, image_dir: str) -> list:
        """
        Predict identities for all images in a directory.

        Args:
            image_dir: Directory containing face images

        Returns:
            list of dicts with filename, identity, confidence
        """
        results = []
        files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])

        for f in files:
            path = os.path.join(image_dir, f)
            try:
                pred = self.predict(path)
                results.append({
                    'filename': f,
                    'identity': pred['identity'],
                    'confidence': pred['confidence'],
                })
            except Exception as e:
                results.append({
                    'filename': f,
                    'identity': 'ERROR',
                    'confidence': 0.0,
                    'error': str(e),
                })

        return results

    @torch.no_grad()
    def evaluate_anonymisation(self, original_dir: str,
                                anonymised_dir: str) -> dict:
        """
        Evaluate anonymisation quality by comparing recognition results
        on original vs anonymised images.

        Both directories should have matching filenames.

        Returns:
            dict with identity_match_rate, per_image_results
        """
        original_preds = self.batch_predict(original_dir)
        anon_preds = self.batch_predict(anonymised_dir)

        # Match by filename
        orig_dict = {r['filename']: r for r in original_preds}
        anon_dict = {r['filename']: r for r in anon_preds}

        common_files = set(orig_dict.keys()) & set(anon_dict.keys())
        matches = 0
        per_image = []

        for f in sorted(common_files):
            orig_id = orig_dict[f]['identity']
            anon_id = anon_dict[f]['identity']
            is_match = (orig_id == anon_id)
            if is_match:
                matches += 1

            per_image.append({
                'filename': f,
                'original_identity': orig_id,
                'anonymised_identity': anon_id,
                'identity_preserved': is_match,
                'original_confidence': orig_dict[f]['confidence'],
                'anonymised_confidence': anon_dict[f]['confidence'],
            })

        total = len(common_files) if common_files else 1
        identity_match_rate = matches / total * 100

        return {
            'identity_match_rate': identity_match_rate,
            'privacy_protection_rate': 100 - identity_match_rate,
            'total_compared': len(common_files),
            'identities_matched': matches,
            'per_image_results': per_image,
            'interpretation': (
                f"The anonymisation {'FAILED' if identity_match_rate > 50 else 'SUCCEEDED'} "
                f"in protecting identity. {identity_match_rate:.1f}% of anonymised images "
                f"were still correctly identified."
            ),
        }


def main():
    parser = argparse.ArgumentParser(description="Face Recognition Inference")
    parser.add_argument("--image", type=str, help="Path to a single face image")
    parser.add_argument("--image1", type=str, help="First image for verification")
    parser.add_argument("--image2", type=str, help="Second image for verification")
    parser.add_argument("--input_dir", type=str, help="Directory for batch prediction")
    parser.add_argument("--output", type=str, default="predictions.csv",
                        help="Output CSV file for batch predictions")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    args = parser.parse_args()

    recognizer = FaceRecognizer(args.checkpoint)

    # Single image prediction
    if args.image:
        result = recognizer.predict(args.image)
        print(f"\nPrediction for: {args.image}")
        print(f"  Identity: {result['identity']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Top-5 predictions:")
        for p in result['top_k_predictions']:
            print(f"    {p['identity']}: {p['confidence']:.4f}")

    # Verification
    elif args.image1 and args.image2:
        result = recognizer.verify(args.image1, args.image2)
        print(f"\nVerification:")
        print(f"  Image 1: {args.image1}")
        print(f"  Image 2: {args.image2}")
        print(f"  Same person: {result['same_person']}")
        print(f"  Similarity: {result['similarity']:.4f}")
        print(f"  Threshold: {result['threshold']}")

    # Batch prediction
    elif args.input_dir:
        import pandas as pd
        results = recognizer.batch_predict(args.input_dir)
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"\nBatch predictions saved to {args.output}")
        print(f"Total images processed: {len(results)}")
        print(df.head(10).to_string())

    else:
        print("Please specify --image, --image1/--image2, or --input_dir")
        parser.print_help()


if __name__ == "__main__":
    main()
