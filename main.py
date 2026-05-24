from os import path

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
import torch

from src.model import CVAE, CVAE2
from src.dataset import CelebADataset
from src.train import train_model
from src.utils import generate_faces

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    TRAIN = True
    SELECTED_ATTRIBUTES = [
    "Bald",
    "Black_Hair",
    "Blond_Hair",
    "Eyeglasses",
    "Male"
]

    train_dataset = CelebADataset(split="train", SELECTED_ATTRIBUTES=SELECTED_ATTRIBUTES, image_size=64)
    print(f"Dataset size: {len(train_dataset)}")

    train_dataset_subset = Subset(train_dataset, range(60000))
    train_loader = DataLoader(train_dataset_subset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)

    model = CVAE(latent_dim=128, n_classes=len(SELECTED_ATTRIBUTES), image_size=64).to(device)


    if(path.exists("cvae.pth") and TRAIN==False):
        model.load_state_dict(torch.load("cvae.pth"))
        print("Model loaded from cvae.pth")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        print("Starting training...")
        train_model(model, train_loader, optimizer, device, epochs=25)
        torch.save(model.state_dict(), "cvae.pth")
        print("Model saved to cvae.pth")
    
    selected_attrs = {
    "Bald": 1,
    "Black_Hair": 1,
    "Blond_Hair": 1,
    "Eyeglasses": 0,
    "Male": 0
    }
    generate_faces(
        model=model,
        selected_attrs=selected_attrs,
        attribute_names=SELECTED_ATTRIBUTES,
        latent_dim=128,
        n_samples=16,
        device=device,
        filename="male_glasses_blackhair.png"
    )   
    




    