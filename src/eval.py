import os
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

from src.train import evaluate_model


def compute_reconstruction(model, data_loader, device):
    """Compute reconstruction metrics (wrapper around evaluate_model)."""
    return evaluate_model(model, data_loader, device)


def compute_latent_tsne(model, data_loader, attribute_names, device, out_path="tsne.png", max_samples=1000):
    """Collect latent means (mu) for samples, run t-SNE, and save a plot.

    Colors are assigned using the first attribute in `attribute_names` for a simple visualization.
    """
    model.eval()

    zs = []
    labels = []

    with torch.no_grad():
        seen = 0
        for x, attrs, _ in data_loader:
            x = x.to(device)
            attrs = attrs.to(device)

            mu, logvar = model.encode(x, attrs)

            zs.append(mu.cpu().numpy())
            # use first attribute as color label if available
            if attrs.size(1) > 0:
                labels.append(attrs[:, 0].cpu().numpy())
            else:
                labels.append(np.zeros((x.size(0),)))

            seen += x.size(0)
            if seen >= max_samples:
                break

    if len(zs) == 0:
        raise RuntimeError("No data collected for t-SNE")

    zs = np.vstack(zs)
    labels = np.hstack(labels)

    tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=42)
    proj = tsne.fit_transform(zs)

    plt.figure(figsize=(8, 8))
    plt.scatter(proj[:, 0], proj[:, 1], c=labels, cmap="coolwarm", s=5)
    plt.title("t-SNE of latent means (mu)")
    plt.colorbar()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path


def compute_fid(fake_dir, real_dir):
    """Compute FID between two folders using pytorch-fid if available.

    If `pytorch_fid` is not installed, raises ImportError with instructions.
    """
    try:
        from pytorch_fid import fid_score
    except Exception as e:
        raise ImportError(
            "pytorch-fid is not installed. Install with: pip install pytorch-fid\n"
            "Or use the provided 'requirements.txt' to install optional deps."
        ) from e

    fid_value = fid_score.calculate_fid_given_paths([real_dir, fake_dir], 50, None, 2048)
    return fid_value
