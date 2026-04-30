import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import DDPMScheduler
from conditional_ddpm import ConditionalUNet
from landmark import get_landmark_and_masked_image
from PIL import Image
import os
from tqdm import tqdm


BATCH_SIZE = 4         
IMG_SIZE = 256
EPOCHS = 30          
LEARNING_RATE = 1e-4
DEVICE = "cuda"


class CelebAHQDataset(Dataset):
    def __init__(self, root_dir="celeba_hq_256"):
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) 
                     if f.endswith(('.jpg', '.png'))]
        self.files = self.files[:1000]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        print("预处理 landmark 中...")
        self.data = []
        for path in tqdm(self.files):
            orig_pil = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            landmark, masked = get_landmark_and_masked_image(path, IMG_SIZE)
            if landmark is not None:
                self.data.append((
                    self.transform(orig_pil),
                    self.transform(landmark),
                    self.transform(masked)
                ))
        print(f"有效图片: {len(self.data)} 张")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

model = ConditionalUNet(img_size=IMG_SIZE).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = DDPMScheduler(num_train_timesteps=1000)

dataset = CelebAHQDataset()
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)


best_val_loss = float('inf')
patience = 5        
no_improve = 0
print(f"=== 开始训练 (共 {EPOCHS} Epochs) ===")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for orig, landmark, masked in tqdm(dataloader):
        orig, landmark, masked = orig.to(DEVICE), landmark.to(DEVICE), masked.to(DEVICE)
        
        noise = torch.randn_like(orig)
        timesteps = torch.randint(0, 1000, (orig.shape[0],)).long().to(DEVICE)
        noisy = scheduler.add_noise(orig, noise, timesteps)
        
        pred_noise = model(noisy, timesteps, landmark, masked)
        loss = nn.functional.mse_loss(pred_noise, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for orig, landmark, masked in val_loader:
            orig, landmark, masked = orig.to(DEVICE), landmark.to(DEVICE), masked.to(DEVICE)
            noise = torch.randn_like(orig)
            timesteps = torch.randint(0, 1000, (orig.shape[0],)).long().to(DEVICE)
            noisy = scheduler.add_noise(orig, noise, timesteps)
            pred_noise = model(noisy, timesteps, landmark, masked)
            val_loss += nn.functional.mse_loss(pred_noise, noise).item()
    
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch:2d}  Train Loss: {avg_loss:.4f}  Val Loss: {avg_val_loss:.4f}")
    
    # Early Stop
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        no_improve = 0

        torch.save(model.state_dict(), "checkpoints/face_anonymizer_best.pth")
        print(f" Val Loss 改善，保存模型")
    else:
        no_improve += 1
        print(f" Val Loss 未改善 ({no_improve}/{patience})")
        if no_improve >= patience:
            print(f" 连续{patience}次未改善，提前停止训练")
            break

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/face_anonymizer_final.pth")

print("checkpoints/face_anonymizer_final.pth")