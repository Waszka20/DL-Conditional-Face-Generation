import os
import time
import json
import torch
from torch import nn


def vae_loss(x_hat, x, mu, logvar, beta=0.1):

    recon = nn.functional.mse_loss(
        x_hat,
        x,
        reduction='mean'
    )

    kl = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return recon + beta * kl, recon, kl


def evaluate_model(model, data_loader, device):
    """Compute reconstruction and KL loss on a data loader (no grad)."""
    model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n = 0

    with torch.no_grad():
        for x, attributes, _ in data_loader:
            x = x.to(device)
            attributes = attributes.to(device)

            x_hat, mu, logvar = model(x, attributes)

            loss, recon, kl = vae_loss(x_hat, x, mu, logvar)

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_recon += recon.item() * batch_size
            total_kl += kl.item() * batch_size
            n += batch_size

    if n == 0:
        return {
            "loss": None,
            "recon": None,
            "kl": None
        }

    return {
        "loss": total_loss / n,
        "recon": total_recon / n,
        "kl": total_kl / n
    }


def train_model(
    model,
    train_loader,
    optimizer,
    device,
    epochs=10,
    val_loader=None,
    checkpoint_dir="checkpoints",
    checkpoint_every=1,
    beta=0.1,
):
    """Train VAE model with optional validation and checkpointing.

    Returns training history dict containing train/val losses per epoch.
    """

    os.makedirs(checkpoint_dir, exist_ok=True)

    history = {
        "train_loss": [],
        "train_recon": [],
        "train_kl": [],
        "val_loss": [],
        "val_recon": [],
        "val_kl": []
    }

    best_val = float("inf")
    all_time = time.time()

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n = 0
        start = time.time()

        for x, attributes, _ in train_loader:
            x = x.to(device)
            attributes = attributes.to(device)

            optimizer.zero_grad()

            x_hat, mu, logvar = model(x, attributes)

            loss, recon, kl = vae_loss(x_hat, x, mu, logvar, beta=beta)

            loss.backward()

            optimizer.step()

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_recon += recon.item() * batch_size
            total_kl += kl.item() * batch_size
            n += batch_size

        # averages
        train_loss = total_loss / max(1, n)
        train_recon = total_recon / max(1, n)
        train_kl = total_kl / max(1, n)

        history["train_loss"].append(train_loss)
        history["train_recon"].append(train_recon)
        history["train_kl"].append(train_kl)

        val_metrics = None
        if val_loader is not None:
            val_metrics = evaluate_model(model, val_loader, device)
            history["val_loss"].append(val_metrics["loss"]) 
            history["val_recon"].append(val_metrics["recon"]) 
            history["val_kl"].append(val_metrics["kl"]) 
        else:
            history["val_loss"].append(None)
            history["val_recon"].append(None)
            history["val_kl"].append(None)

        end = time.time()

        print(
            f"Epoch {epoch+1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Recon: {train_recon:.4f} | "
            f"KL: {train_kl:.4f} | "
            f"Time: {end - start:.2f}s"
        )

        if val_metrics is not None:
            print(
                f"  Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Recon: {val_metrics['recon']:.4f} | "
                f"Val KL: {val_metrics['kl']:.4f}"
            )

            # checkpoint best
            if val_metrics["loss"] is not None and val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                ckpt_path = os.path.join(checkpoint_dir, "best.pth")
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "history": history
                }, ckpt_path)
                print(f"Saved best checkpoint to: {ckpt_path}")

        # periodic checkpoint
        if (epoch + 1) % checkpoint_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "history": history
            }, ckpt_path)

    print(f"Total training time: {time.time() - all_time:.2f}s")

    # save history
    hist_path = os.path.join(checkpoint_dir, "history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    return history