import os
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
from PIL import Image
import torch.nn.functional as F

def make_attribute_vector(selected_attrs, attribute_names):
    """
    selected_attrs:
    {
        "Male": 1,
        "Eyeglasses": 1
    }
    """

    return torch.tensor(
        [selected_attrs.get(attr, 0)for attr in attribute_names]
        ,dtype=torch.float32)


def generate_faces(
    model,
    selected_attrs,
    attribute_names,
    latent_dim=64,
    n_samples=16,
    device="cuda",
    save_dir="generated",
    filename="generated_faces.png"
):

    model.eval()

    os.makedirs(save_dir, exist_ok=True)

    attributes = make_attribute_vector(
        selected_attrs,
        attribute_names
    )

    with torch.no_grad():

        # sample latent vectors
        z = torch.randn(
            n_samples,
            latent_dim
        ).to(device)

        # repeat attributes for batch
        attributes = attributes.unsqueeze(0).repeat(
            n_samples,
            1
        )

        attributes = attributes.to(device)

        # generate
        samples = model.decode(z, attributes)

        # [-1,1] -> [0,1]
        samples = (samples + 1) / 2

        samples = torch.clamp(samples, 0, 1)

        # create grid
        grid = make_grid(
            samples,
            nrow=4
        )

        save_path = os.path.join(
            save_dir,
            filename
        )

        save_image(grid, save_path)

        print(f"Generated images saved to: {save_path}")

    return samples


def make_reconstructions(
    model,
    data_loader,
    save_path,
    n_images=8,
    device="cuda"
):
    model.eval()

    imgs, attrs, _ = next(iter(data_loader))

    imgs = imgs[:n_images].to(device)
    attrs = attrs[:n_images].to(device)

    with torch.no_grad():

        mu, logvar = model.encode(
            imgs,
            attrs
        )

        recon = model.decode(
            mu,
            attrs
        )

    originals = ((imgs.cpu() + 1) / 2).clamp(0, 1)
    reconstructions = ((recon.cpu() + 1) / 2).clamp(0, 1)

    fig, axes = plt.subplots(
        2,
        n_images,
        figsize=(2*n_images, 4)
    )

    for i in range(n_images):

        axes[0, i].imshow(
            originals[i].permute(1, 2, 0)
        )
        axes[0, i].axis("off")

        axes[1, i].imshow(
            reconstructions[i].permute(1, 2, 0)
        )
        axes[1, i].axis("off")

    axes[0, 0].set_title("Original")
    axes[1, 0].set_title("Reconstruction")

    plt.tight_layout()
    plt.savefig(
        save_path,
        bbox_inches="tight"
    )

    plt.close()


def show_generated_faces(
    model,
    selected_attrs,
    attribute_names,
    latent_dim=64,
    n_samples=16,
    device="cuda"
):

    model.eval()

    attributes = make_attribute_vector(
        selected_attrs,
        attribute_names
    )

    with torch.no_grad():

        z = torch.randn(
            n_samples,
            latent_dim
        ).to(device)

        attributes = attributes.unsqueeze(0).repeat(
            n_samples,
            1
        )

        attributes = attributes.to(device)

        samples = model.decode(z, attributes)

        # [-1,1] -> [0,1]
        samples = (samples + 1) / 2

        samples = torch.clamp(samples, 0, 1)

        grid = make_grid(
            samples,
            nrow=4
        )

        plt.figure(figsize=(8, 8))

        plt.imshow(
            grid.permute(1, 2, 0).cpu().numpy()
        )

        plt.axis("off")

        plt.show()

def load_image(image_path, image_size=64):

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    return image.unsqueeze(0)


import matplotlib.pyplot as plt


def show_tensor_image(img):

    img = img.squeeze(0)

    img = (img + 1) / 2

    img = torch.clamp(img, 0, 1)

    plt.figure(figsize=(4,4))

    plt.imshow(
        img.permute(1,2,0).cpu().numpy()
    )

    plt.axis("off")

    plt.show()


def edit_face_attributes(
    model,
    image,
    new_attrs,
    attribute_names,
    device="cuda"
):

    model.eval()

    with torch.no_grad():

        x = image.to(device)

        new_y = make_attribute_vector(
            new_attrs,
            attribute_names
        ).unsqueeze(0).to(device)

        # encode original image
        mu, logvar = model.encode(x, new_y)

        # use mean latent vector
        z = mu

        # decode with NEW attributes
        edited = model.decode(z, new_y)

    return edited



def show_from_latent(
    model,
    z,
    attrs,
    attribute_names,
    device="cuda"
):
    """
    Decode a given latent vector z with given attributes
    and display the result.
    """

    model.eval()

    with torch.no_grad():

        if z.dim() == 1:
            z = z.unsqueeze(0)

        z = z.to(device)

        y = make_attribute_vector(
            attrs,
            attribute_names
        ).unsqueeze(0).to(device)

        img = model.decode(z, y)

        img = (img + 1) / 2
        img = torch.clamp(img, 0, 1)

    plt.figure(figsize=(4, 4))

    plt.imshow(
        img.squeeze(0)
           .permute(1, 2, 0)
           .cpu()
           .numpy()
    )

    plt.axis("off")
    plt.show()


def analyze_reconstruction(
    model,
    image,
    attrs,
    attribute_names,
    device="cuda"
):
    """
    Encode -> Decode analysis.

    Shows:
    - original image
    - reconstructed image
    - absolute error map

    Prints:
    - MSE
    - MAE
    """

    model.eval()

    x = image.unsqueeze(0).to(device)

    y = make_attribute_vector(
        attrs,
        attribute_names
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        mu, logvar = model.encode(x, y)

        z = mu

        x_hat = model.decode(z, y)

    mse = F.mse_loss(x_hat, x).item()
    mae = F.l1_loss(x_hat, x).item()

    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")

    original = (x.squeeze(0).cpu() + 1) / 2
    reconstructed = (x_hat.squeeze(0).cpu() + 1) / 2

    original = torch.clamp(original, 0, 1)
    reconstructed = torch.clamp(reconstructed, 0, 1)

    error = torch.abs(
        reconstructed - original
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12,4)
    )

    axes[0].imshow(
        original.permute(1,2,0)
    )
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(
        reconstructed.permute(1,2,0)
    )
    axes[1].set_title("Reconstructed")
    axes[1].axis("off")


    plt.tight_layout()
    plt.show()

    return {
        "mse": mse,
        "mae": mae,
        "mu": mu.cpu(),
        "logvar": logvar.cpu()
    }




def show_attribute_transition(
    model,
    image,
    start_attrs,
    end_attrs,
    attribute_names,
    steps=6,
    device="cuda"
):

    model.eval()

    with torch.no_grad():

        x = image.unsqueeze(0).to(device)

        start_y = make_attribute_vector(
            start_attrs,
            attribute_names
        ).to(device)

        end_y = make_attribute_vector(
            end_attrs,
            attribute_names
        ).to(device)

        #
        # encode only once
        #
        mu, logvar = model.encode(
            x,
            start_y.unsqueeze(0)
        )

        z = mu

        images = []

        for alpha in torch.linspace(0, 1, steps):

            y = (
                (1 - alpha) * start_y
                + alpha * end_y
            )

            y = y.unsqueeze(0)

            img = model.decode(
                z,
                y
            )

            img = img.squeeze(0).cpu()

            img = (img + 1) / 2
            img = torch.clamp(img, 0, 1)

            images.append(img)

    #
    # show results
    #
    fig, axes = plt.subplots(
        1,
        steps,
        figsize=(3 * steps, 3)
    )

    for i in range(steps):

        axes[i].imshow(
            images[i].permute(1, 2, 0)
        )

        axes[i].set_title(
            f"{i/(steps-1):.2f}"
        )

        axes[i].axis("off")

    plt.tight_layout()
    plt.show()