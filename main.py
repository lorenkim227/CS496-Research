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

# densenet121
def get_densenet121(num_classes_binary=2, num_classes_multi=4):
    model = models.densenet121(pretrained=True)
    in_features = model.classifier.in_features

    model.classifier = nn.Identity()

    classifier_binary = nn.Linear(in_features, num_classes_binary)
    classifier_multi = nn.Linear(in_features, num_classes_multi)

    return model, classifier_binary, classifier_multi

# efficientnet_b3
def get_efficientnet_b3(num_classes_binary=2, num_classes_multi=4):
    model = models.efficientnet_b3(pretrained=True)
    in_features = model.classifier[1].in_features

    model.classifier = nn.Identity()

    classifier_binary = nn.Linear(in_features, num_classes_binary)
    classifier_multi = nn.Linear(in_features, num_classes_multi)

    return model, classifier_binary, classifier_multi


# training
def train_model(backbone, clf_bin, clf_multi, train_loader, val_loader, epochs=10):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone.to(device)
    clf_bin.to(device)
    clf_multi.to(device)

    optimizer = optim.Adam(
        list(backbone.parameters()) +
        list(clf_bin.parameters()) +
        list(clf_multi.parameters()),
        lr=1e-4
    )

    criterion_bin = nn.CrossEntropyLoss()
    criterion_multi = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        backbone.train()
        clf_bin.train()
        clf_multi.train()

        total_loss = 0

        for images, fracture, fracture_type in tqdm(train_loader):
            images, fracture, fracture_type = images.to(device), fracture.to(device), fracture_type.to(device)

            optimizer.zero_grad()

            features = backbone(images)

            out_bin = clf_bin(features)
            out_multi = clf_multi(features)

            loss_bin = criterion_bin(out_bin, fracture)

            # compute multi-class loss only if fracture = 1
            mask = fracture == 1
            if mask.sum() > 0:
                loss_multi = criterion_multi(out_multi[mask], fracture_type[mask])
            else:
                loss_multi = 0

            loss = loss_bin + loss_multi
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


# evaluation
def evaluate_model(backbone, clf_bin, clf_multi, loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone.eval()
    clf_bin.eval()
    clf_multi.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, fracture, fracture_type in loader:
            images = images.to(device)

            features = backbone(images)
            out_bin = clf_bin(features)

            preds = torch.argmax(out_bin, dim=1).cpu().numpy()

            y_true.extend(fracture.numpy())
            y_pred.extend(preds)

    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))



