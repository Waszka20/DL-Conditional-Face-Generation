import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms
import os
import kagglehub

class CelebADataset(Dataset):
    def __init__(self, split="train", SELECTED_ATTRIBUTES=None, image_size=128):
        self.path = kagglehub.dataset_download("jessicali9530/celeba-dataset")
        self.image_path = os.path.join(self.path, "img_align_celeba", "img_align_celeba")
        self.image_size = image_size

        attrs = pd.read_csv(os.path.join(self.path, "list_attr_celeba.csv"))
        
        attrs = attrs.set_index("image_id")
        attrs = attrs[SELECTED_ATTRIBUTES]
        self.attributes = attrs
        self.attributes_names = SELECTED_ATTRIBUTES

        split_df = pd.read_csv(os.path.join(self.path, "list_eval_partition.csv"))
        
        partition_map = {
            "train": 0,
            "eval": 1,
            "test": 2
        }
        
        self.images = split_df[split_df["partition"] == partition_map[split]]["image_id"].values
        
        self.transform = transforms.Compose([
            #Originally 218x178, resize to 128x128 for faster training
            #transforms.CenterCrop((178, 178)),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_id = self.images[idx]
        
        image = Image.open(os.path.join(self.image_path, image_id)).convert("RGB")
        image = self.transform(image)
        
        attrs = self.attributes.loc[image_id].values
        attrs = torch.tensor((attrs + 1) // 2, dtype=torch.float32) # wartosci 0,1
        
        return image, attrs, image_id