# Face Anonymization with Conditional UNet

This project implements a face anonymization pipeline using a Conditional UNet and a diffusion-based denoising process. The model generates anonymous faces while preserving the structural integrity of the original image, ensuring privacy protection while maintaining visual realism.

A Pre-trained anonymized model can be download in Github Releases.

## Features
- **Face Anonymization**: Generates anonymous faces that retain the structure of the original face but alter the identity.
- **Landmark and Mask Extraction**: Extracts facial landmarks and masks to guide the anonymization process.
- **Diffusion Model**: Uses a DDIMScheduler for iterative denoising, ensuring high-quality outputs.
- **Batch Processing**: Processes multiple images in a batch and saves the anonymized results.

## Requirements
To run this project, you need the following dependencies:
- Python 3.8+
- PyTorch
- torchvision
- OpenCV
- Pillow
- numpy
- mediapipe
- diffusers

Install the required packages using pip:
```bash
pip install torch torchvision opencv-python pillow numpy mediapipe diffusers
```

## File Structure
```
.
├── download_dataset.py    # Script to download the dataset
├── conditional_ddpm.py    # Conditional UNet model definition
├── landmark.py            # Landmark and mask extraction
├── inference.py           # Main script for face anonymization
├── train.py               # Training script for the model
├── checkpoints/           # Folder containing the trained model weights
├── test_img/              # Folder containing test images
├── results/               # Folder to save anonymized images and debug outputs
└── README.md              # Project documentation
```

## Usage

### 1. Download the Dataset（the dataset has already been download）
Run the `download_dataset.py` script to download the CelebA-HQ dataset:
```bash
python download_dataset.py
```

### 2. Train the Model
To train the model, run the `train.py` script:
```bash
python train.py
```
The trained model weights will be saved in the `checkpoints/` folder.

### 3. Run the Anonymization Script
Run the `inference.py` script to anonymize the images:
```bash
python inference.py
```
The script will:
1. Process images from `test_img/`.
2. Save the anonymized images in the `results/batch/` folder.
3. Save intermediate debug outputs (e.g., noisy images, predicted noise) in the `results/` folder.

## How It Works

1. **Landmark and Mask Extraction**:
   - The `get_landmark_and_masked_image` function extracts facial landmarks and creates a masked image with the face region blacked out.

2. **Noise Addition**:
   - Random noise is added to the original image at a specific timestep (`t_start=500`).

3. **Iterative Denoising**:
   - The Conditional UNet predicts the noise at each timestep.
   - The DDIMScheduler removes the predicted noise iteratively, generating a clean image.

4. **Face Fusion**:
   - The generated anonymous face is blended with the original image using OpenCV's `seamlessClone` to preserve the background and non-facial features.

## Debugging
Intermediate results are saved in the `results/` folder:
- `debug_mask.jpg`: The masked image.
- `debug_landmark.jpg`: The landmark visualization.
- `debug_noisy.jpg`: The noisy image at the start of the denoising process.
- `debug_predicted_noise_t{t}.jpg`: The predicted noise at each timestep.

## Acknowledgments
This project uses the following libraries and tools:
- [PyTorch](https://pytorch.org/)
- [Mediapipe](https://google.github.io/mediapipe/)
- [Diffusers](https://huggingface.co/docs/diffusers/)
- [OpenCV](https://opencv.org/)

## License
This project is for educational purposes only. Redistribution or commercial use is not permitted without prior permission.
