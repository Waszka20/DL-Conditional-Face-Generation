import torch
from torch import nn
import time

def vae_loss(x_hat, x, mu, logvar, beta=1):

    recon = nn.functional.mse_loss(
        x_hat,
        x,
        reduction='mean'
    )

    kl = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return recon + beta * kl, recon, kl


def train_model(
    model,
    train_loader,
    optimizer,
    device,
    epochs=10
):
   
    all_time = time.time()
    for epoch in range(epochs):

        model.train()

        total_loss = 0
        total_recon = 0
        total_kl = 0
        start = time.time()

        for x, attributes, _ in train_loader:

            x = x.to(device)
            attributes = attributes.to(device)

            optimizer.zero_grad()

            x_hat, mu, logvar = model(x, attributes)

            loss, recon, kl = vae_loss(
                x_hat,
                x,
                mu,
                logvar
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
        end = time.time()

        print(
            f"Epoch {epoch+1} | "
            f"Loss: {total_loss / len(train_loader):.4f} | "
            f"Recon: {total_recon / len(train_loader):.4f} | "
            f"KL: {total_kl / len(train_loader):.4f} | "
            f"Time: {end - start:.2f}s"
        )
    print(f"Total training time: {time.time() - all_time:.2f}s")