from landmark import get_landmark_and_masked_image
import matplotlib.pyplot as plt
from PIL import Image
import os

test_images = [
    "celeba_hq_256/00001.jpg",
    "celeba_hq_256/00002.jpg",
    "celeba_hq_256/00003.jpg",
    "celeba_hq_256/00004.jpg",
    "celeba_hq_256/00005.jpg",
]

fig, axes = plt.subplots(len(test_images), 3, figsize=(12, 4 * len(test_images)))

for i, path in enumerate(test_images):
    landmark, masked = get_landmark_and_masked_image(path, output_size=256)
    orig = Image.open(path).convert("RGB").resize((256, 256))
    
    if landmark is None:
        print(f"{path} 未检测到人脸")
        continue
    else:
        print(f"{path} 检测成功")

    axes[i][0].imshow(orig)
    axes[i][0].set_title(f"原图 {i+1}")
    axes[i][1].imshow(landmark)
    axes[i][1].set_title("Landmark")
    axes[i][2].imshow(masked)
    axes[i][2].set_title("Masked")

    for ax in axes[i]:
        ax.axis("off")

plt.tight_layout()
plt.savefig("results/landmark_test_multi.png")
plt.show()