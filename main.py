# Loren Kim
# main.py for CS496 Research

import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

# import dataset
class FractureDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        image = Image.open(img_path).convert("RGB")

        fracture = self.df.iloc[idx]['fracture']
        fracture_type = self.df.iloc[idx]['fracture_type']

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(fracture).long(), torch.tensor(fracture_type).long()
    

# data preprocessing
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


# test train split
df = pd.read_csv("data.csv")

train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['fracture'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['fracture'], random_state=42)

train_dataset = FractureDataset(train_df, transform=train_transform)
val_dataset = FractureDataset(val_df, transform=val_transform)
test_dataset = FractureDataset(test_df, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)


# resnet50
def get_resnet50(num_classes_binary=2, num_classes_multi=4):
    model = models.resnet50(pretrained=True)
    in_features = model.fc.in_features

    model.fc = nn.Identity()

    classifier_binary = nn.Linear(in_features, num_classes_binary)
    classifier_multi = nn.Linear(in_features, num_classes_multi)

    return model, classifier_binary, classifier_multi

def get_model(model_name):
    if model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "densenet121":
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif model_name == "efficientnet_b3":
        model = models.efficientnet_b3(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model.to(device)