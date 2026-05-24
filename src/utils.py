import os
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
from PIL import Image

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
    original_attrs,
    new_attrs,
    attribute_names,
    device="cuda"
):

    model.eval()

    with torch.no_grad():

        x = image.to(device)

        original_y = make_attribute_vector(
            original_attrs,
            attribute_names
        ).unsqueeze(0).to(device)

        new_y = make_attribute_vector(
            new_attrs,
            attribute_names
        ).unsqueeze(0).to(device)

        # encode original image
        mu, logvar = model.encode(x, original_y)

        # use mean latent vector
        z = mu

        # decode with NEW attributes
        edited = model.decode(z, new_y)

    return edited