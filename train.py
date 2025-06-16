
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from utils import RoofDataset, get_training_augmentation, get_validation_augmentation, dice_score
from tqdm import tqdm

# Diretórios
DATA_DIR = 'data/train'
MASK_DIR = 'data/train_masks'
MODEL_DIR = 'models'

# Hiperparâmetros
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
IMG_SIZE = 256

# Dataset e DataLoader
train_dataset = RoofDataset(
    images_dir=DATA_DIR,
    masks_dir=MASK_DIR,
    augmentation=get_training_augmentation(IMG_SIZE)
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Modelo
model = smp.FPN(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Otimizador e função de perda
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.BCEWithLogitsLoss()

# Treinamento
model.train()
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device).unsqueeze(1)

        preds = model(images)
        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} Loss: {epoch_loss/len(train_loader):.4f}")

# Salvando o modelo
os.makedirs(MODEL_DIR, exist_ok=True)
torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'fpn_resnet34.pth'))
print("\nModelo salvo com sucesso!")
