
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from utils import RoofDataset, get_validation_augmentation, dice_score
from torchvision.utils import save_image

# Diretórios
IMAGE_DIR = 'data/test'
MASK_DIR = 'data/test_masks'
MODEL_PATH = 'models/fpn_resnet34.pth'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Parâmetros
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
BATCH_SIZE = 8

# Dataset e DataLoader
test_dataset = RoofDataset(
    images_dir=IMAGE_DIR,
    masks_dir=MASK_DIR,
    augmentation=get_validation_augmentation(IMG_SIZE)
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Modelo
model = smp.FPN(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Avaliação e salvamento
all_dice_scores = []
with torch.no_grad():
    for i, (images, masks) in enumerate(test_loader):
        images = images.to(device)
        masks = masks.to(device).unsqueeze(1)

        outputs = model(images)
        preds = torch.sigmoid(outputs) > 0.5

        # Dice score
        dice = dice_score(preds.float(), masks.float())
        all_dice_scores.append(dice.item())

        # Salvando resultados
        for j in range(images.size(0)):
            result_path = os.path.join(RESULTS_DIR, f'result_{i*BATCH_SIZE + j}.png')
            save_image(preds[j].float(), result_path)

# Resultado final
mean_dice = np.mean(all_dice_scores)
print(f"\nDice Score médio no conjunto de teste: {mean_dice:.4f}")
