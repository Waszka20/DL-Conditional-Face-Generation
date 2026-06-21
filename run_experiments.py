import argparse
import os
import torch
from torch.utils.data import DataLoader, Subset

from src.dataset import CelebADataset
from src.model import CVAE, CVAE2
from src.train import train_model
from src.eval import compute_latent_tsne


def make_dataloaders(selected_attributes, image_size=64, batch_size=128, max_train=10000, max_val=2000):
    train_ds = CelebADataset(split="train", SELECTED_ATTRIBUTES=selected_attributes, image_size=image_size)
    val_ds = CelebADataset(split="eval", SELECTED_ATTRIBUTES=selected_attributes, image_size=image_size)

    train_subset = Subset(train_ds, range(min(len(train_ds), max_train)))
    val_subset = Subset(val_ds, range(min(len(val_ds), max_val)))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SELECTED_ATTRIBUTES = args.attributes

    train_loader, val_loader = make_dataloaders(SELECTED_ATTRIBUTES, image_size=args.image_size, batch_size=args.batch_size, max_train=args.max_train, max_val=args.max_val)

    # CVAE (conv)
    cvae = CVAE(latent_dim=args.latent_dim, n_classes=len(SELECTED_ATTRIBUTES), image_size=args.image_size).to(device)
    optimizer = torch.optim.Adam(cvae.parameters(), lr=args.lr)
    print("Training CVAE (conv)...")
    train_model(cvae, train_loader, optimizer, device, epochs=args.epochs, val_loader=val_loader, checkpoint_dir=os.path.join(args.out_dir, "cvae_ckpts"), checkpoint_every=1)
    torch.save(cvae.state_dict(), os.path.join(args.out_dir, "cvae_final.pth"))

    # CVAE2 (fc)
    cvae2 = CVAE2(latent_dim=args.latent_dim, n_classes=len(SELECTED_ATTRIBUTES), image_size=args.image_size).to(device)
    optimizer2 = torch.optim.Adam(cvae2.parameters(), lr=args.lr)
    print("Training CVAE2 (fc)...")
    train_model(cvae2, train_loader, optimizer2, device, epochs=args.epochs, val_loader=val_loader, checkpoint_dir=os.path.join(args.out_dir, "cvae2_ckpts"), checkpoint_every=1)
    torch.save(cvae2.state_dict(), os.path.join(args.out_dir, "cvae2_final.pth"))

    # latent visualization for CVAE
    print("Computing latent visualization for CVAE...")
    os.makedirs(args.out_dir, exist_ok=True)
    tsne_path = os.path.join(args.out_dir, "cvae_tsne.png")
    compute_latent_tsne(cvae, val_loader, SELECTED_ATTRIBUTES, device, out_path=tsne_path)
    print(f"Saved t-SNE plot to: {tsne_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="experiments", help="output dir for checkpoints and artifacts")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-train", type=int, default=2000)
    parser.add_argument("--max-val", type=int, default=500)
    parser.add_argument("--attributes", nargs="*", default=["Bald","Black_Hair","Blond_Hair","Eyeglasses","Male"]) 

    args = parser.parse_args()

    run(args)
