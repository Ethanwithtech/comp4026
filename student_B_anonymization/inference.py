import torch
from PIL import Image
import cv2
import numpy as np
from torchvision import transforms
from conditional_ddpm import ConditionalUNet
from landmark import get_landmark_and_masked_image

import os

from diffusers import DDIMScheduler         

scheduler = DDIMScheduler(num_train_timesteps=1000)
scheduler.set_timesteps(50)

DEVICE = "cuda"
IMG_SIZE = 256

model = ConditionalUNet(img_size=IMG_SIZE).to(DEVICE)
model.load_state_dict(torch.load("checkpoints/face_anonymizer_final.pth", map_location=DEVICE))
model.eval()

def generate_anonymized(image_path, output_path="results/anonymized.jpg"):
    orig_pil = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))

    landmark, masked = get_landmark_and_masked_image(image_path, IMG_SIZE)
    
    if landmark is None:
        print("未检测到人脸")
        return
    
    masked_pil = masked  


    masked_pil.save("results/debug_mask.jpg")
    landmark.save("results/debug_landmark.jpg") 

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    orig = to_tensor(orig_pil).unsqueeze(0).to(DEVICE)
    landmark = to_tensor(landmark).unsqueeze(0).to(DEVICE)
    masked = to_tensor(masked_pil).unsqueeze(0).to(DEVICE)


    t_start = 500 
    noise = torch.randn_like(orig)
    timesteps_tensor = torch.tensor([t_start]).long().to(DEVICE)
    noisy = scheduler.add_noise(orig, noise, timesteps_tensor)


    noisy_image = noisy.squeeze(0).cpu()
    noisy_image = (noisy_image * 0.5 + 0.5).clamp(0, 1)  
    noisy_pil = transforms.ToPILImage()(noisy_image)
    noisy_pil.save("results/debug_noisy.jpg") 


    scheduler.set_timesteps(50)
    timesteps_to_use = [t for t in scheduler.timesteps if t <= t_start]
    
    for t in timesteps_to_use:
        with torch.no_grad():
            noise_pred = model(noisy, t.to(DEVICE), landmark, masked)
        noisy = scheduler.step(noise_pred, t, noisy).prev_sample

    predicted_noise_image = noise_pred.squeeze(0).cpu()
    predicted_noise_image = (predicted_noise_image * 0.5 + 0.5).clamp(0, 1) 
    predicted_noise_pil = transforms.ToPILImage()(predicted_noise_image)
    predicted_noise_pil.save(f"results/debug_predicted_noise_t{t.item()}.jpg")


    generated = noisy.squeeze(0).cpu()
    generated = (generated * 0.5 + 0.5).clamp(0, 1)
    generated_pil = transforms.ToPILImage()(generated)
    
    orig_cv = cv2.cvtColor(np.array(orig_pil), cv2.COLOR_RGB2BGR)
    gen_cv = cv2.cvtColor(np.array(generated_pil), cv2.COLOR_RGB2BGR)
    
    masked_np = np.array(masked_pil.convert("L"))
    face_mask = (masked_np < 128).astype(np.uint8) * 255 
    
    center = (orig_cv.shape[1] // 2, orig_cv.shape[0] // 2)
    result = cv2.seamlessClone(gen_cv, orig_cv, face_mask, center, cv2.NORMAL_CLONE)
    
    cv2.imwrite(output_path, result)
    print(f"匿名化图片已保存: {output_path}")


if __name__ == "__main__":
    os.makedirs("results/batch", exist_ok=True)
    
    for i in range(10000, 10040):
        image_path = f"test_img/{i:05d}.jpg"
        output_path = f"results/batch/{i:05d}_anonymized.jpg"
        
        if not os.path.exists(image_path):
            print(f"图片不存在: {image_path}")
            continue
        
        print(f"处理: {image_path}")
        generate_anonymized(image_path, output_path=output_path)

