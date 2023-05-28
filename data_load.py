import os
import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from PIL import Image

# My modules
from config import config
import numpy as np
import random 


class CustomDataset(Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        self.images_path = images_path
        self.labels_path = labels_path
        self.image_filenames = [os.path.basename(filename).split('.')[0] for filename in os.listdir(images_path)]
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.images_path, (self.image_filenames[index] + ".tiff"))
        label_path = os.path.join(self.labels_path, (self.image_filenames[index] + ".tif"))
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        if self.transform:
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            random.seed(seed)
            transformed_image = self.transform(image).float()
            torch.manual_seed(seed)
            random.seed(seed)
            transformed_label = self.transform(label).float()
            return transformed_image, transformed_label
        else:
            image = to_tensor(image)
            label = to_tensor(label)  
            return image.float(), label.float()


# transforms.Resize((1472, 1472), antialias=True),
transform = {
    "train": transforms.Compose([
            transforms.Pad(padding=2),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor()]),
    "val": transforms.Compose([
            transforms.Pad(padding=2),
            transforms.ToTensor()])
}

def load_dataset():
    train = os.path.join(config["data_dir"], 'train')
    train_labels = os.path.join(config["data_dir"], 'train_labels')

    val = os.path.join(config["data_dir"], 'val')
    val_labels = os.path.join(config["data_dir"], 'val_labels')

    test = os.path.join(config["data_dir"], 'test')
    test_labels = os.path.join(config["data_dir"], 'test_labels')

    train_set = CustomDataset(train, train_labels, transform=transform["train"])
    test_set = CustomDataset(test, test_labels)
    val_set = CustomDataset(val, val_labels, transform=transform["val"])

    loader = {
        "train": DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, pin_memory=True, num_workers=config["num_workers"]),
        "test": DataLoader(test_set, batch_size=config["batch_size"], shuffle=False, pin_memory=True, num_workers=config["num_workers"]),
        "val": DataLoader(val_set, batch_size=config["batch_size"], shuffle=False, pin_memory=True, num_workers=config["num_workers"])
    }

    return loader