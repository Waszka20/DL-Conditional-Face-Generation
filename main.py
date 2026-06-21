import json
from os import path
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
import torch
import shutil

from src.model import CVAE, CVAE2
from src.dataset import CelebADataset
from src.train import train_model
from src.utils import generate_faces,make_reconstructions
from src.config import load_config, build_experiment_name




def make_dataloaders(selected_attributes, image_size=64, batch_size=128, max_train=10000, max_val=6000):
    train_ds = CelebADataset(split="train", SELECTED_ATTRIBUTES=selected_attributes, image_size=image_size)
    val_ds = CelebADataset(split="eval", SELECTED_ATTRIBUTES=selected_attributes, image_size=image_size)
    test_ds = CelebADataset(split="test", SELECTED_ATTRIBUTES=selected_attributes, image_size=image_size)
    print(f"Dataset size: {len(train_ds)}")

    if(max_train=="all"):
        max_train = len(train_ds)
    train_subset = Subset(train_ds, range(min(len(train_ds), max_train)))
    val_subset = Subset(val_ds, range(min(len(val_ds), max_val)))
    test_subset = Subset(test_ds, range(min(len(test_ds), max_val)))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":

    cfg = load_config()
    experiment_name = build_experiment_name(cfg)


    c_latent_dim = cfg["model"]["latent_dim"]
    c_lr = cfg["training"]["lr"]
    c_beta = cfg["training"]["beta"]
    c_epochs = cfg["training"]["epochs"]
    c_image_size = cfg["model"]["image_size"]
    c_batch_size = cfg["training"]["batch_size"]
    c_train_dataset_size = cfg["dataset"]["train_subset_size"]
    c_attributes = cfg["attributes"]

    print("Experiment:", experiment_name)
    exp_dir = os.path.join(
        "experiments",
        experiment_name
    )
    os.makedirs(
        exp_dir,
        exist_ok=True
    )
    shutil.copy(
    "config.yaml",
    os.path.join(
        exp_dir,
        "config.yaml"
        )
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    TRAIN = True
    CELEBA_ATTRIBUTES = [
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young"
]
    SELECTED_ATTRIBUTES = c_attributes

    train_loader, val_loader, test_loader = make_dataloaders(SELECTED_ATTRIBUTES, image_size=c_image_size, batch_size=c_batch_size, max_train=c_train_dataset_size)

    model = CVAE(latent_dim=c_latent_dim, n_classes=len(SELECTED_ATTRIBUTES), image_size=c_image_size).to(device)

    model_path = os.path.join(exp_dir,"model.pth")
    if path.exists(model_path) and not TRAIN:
        model.load_state_dict(
            torch.load(model_path)
        )
        print(f"Loaded model: {model_path}")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=c_lr)
        print("Starting training...")
        history = train_model(model, train_loader, optimizer, device, epochs=c_epochs, beta=c_beta)
        torch.save(model.state_dict(),  os.path.join(exp_dir,"model.pth"))
        print("Model saved to :", os.path.join(exp_dir,"model.pth"))
        with open(os.path.join(exp_dir, "history.json"), "w") as f:
            json.dump(history, f)
    

    selected_attrs = {
    "Bald": 0,
    "Black_Hair": 1,
    "Blond_Hair": 0,
    "Eyeglasses": 1,
    "Male": 1
    }
    generated_dir = os.path.join(
        exp_dir,
        "generated"
    )

    reconstruction_dir = os.path.join(
        exp_dir,
        "reconstructions"
    )
    os.makedirs(generated_dir, exist_ok=True)
    os.makedirs(reconstruction_dir, exist_ok=True)
    
    generate_faces(
        model=model,
        selected_attrs=selected_attrs,
        attribute_names=SELECTED_ATTRIBUTES,
        latent_dim=c_latent_dim,
        n_samples=16,
        device=device,
        save_dir=generated_dir,
        filename="male_glasses_blackhair.png"
    )   
    
    make_reconstructions(
    model,
    test_loader,
    save_path=os.path.join(
        reconstruction_dir,
        "final_reconstruction.png"
    ),
    n_images=8,
    device=device
)




    