#You need to install segmentation-model-pytorch with this command : pip install segmentation-models-pytorch
#Install Pytorch according this link : #https://pytorch.org/get-started
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from segmentation_models_pytorch import Unet
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset

# Path to the directory . This Directory should have three folder with name of "train" ,  "validation", "test". Each folder should contain two folder "0" , "1".
data_dir = "path to your Directory"
image_size = 512
batch = 2
epochs = 10
print_interval = 10  # Print the loss and IoU every 10 steps
image_format = "png"
encoder_name = "resnet34"
encoder_weights = "imagenet"
input_channel = 3
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None, image_format = "png",image_size = 512):
        self.data_dir = data_dir
        self.transform = transform
        self.image_format = image_format
        self.image_dir = os.path.join(data_dir)
        self.mask_dir = os.path.join(data_dir)
        self.classes_dir = os.listdir(self.image_dir)
        self.image_filenames = sorted([f for f in os.listdir(self.image_dir) if f.endswith(f".{image_format}")])
        self.image_size = image_size
    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir,self.classes_dir,image_name)
        mask_path = os.path.join(self.mask_dir,self.classes_dir, image_name)

        # Open image and mask using PIL
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale (1 channel)
        mask = mask.resize((self.image_size,self.image_size))
        if self.transform:
            image = self.transform(image)

        # Convert mask to binary (0 or 1)
        mask = np.array(mask)  # Convert mask to NumPy array
        mask = np.where(mask > 0, 1, 0)  # Perform comparison and convert to binary

        # Add an additional dimension to the mask
        mask = np.expand_dims(mask, axis=0)

        # Convert the mask to float
        mask = torch.tensor(mask, dtype=torch.float32)

        return image, mask
# Number of classes (binary segmentation: 0 and 1)

# Transformations for data preprocessing
transform = T.Compose([
    #T.RandomResizedCrop((256, 256)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(degrees=15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.Resize((image_size, image_size)),  # Add this line to resize masks to the same size as images
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_transform = T.Compose([
    T.Resize((image_size, image_size)),  # Add this line to resize masks to the same size as images
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Load your custom dataset using DataLoader
train_dataset = CustomDataset(data_dir, transform=transform, image_format=image_format,image_size=image_size)
val_dataset = CustomDataset(data_dir, transform=transform, image_format=image_format,image_size=image_size)
test_dataset = CustomDataset(data_dir, transform=test_transform, image_format=image_format,image_size=image_size)

train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

# Define the model (UNet with ResNet34 backbone) for binary segmentation
model = Unet(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=1, activation=None, in_channels=input_channel)
#-----------Model----------------------
#You can choose one of them for models :Unet ,  UnetPlusPlus ,MAnet,Linknet ,FPN,PSPNet,PAN,DeepLabV3,DeepLabV3+
#-----------Encoder--------------------
#In this link you can use one of them for encoder :https://smp.readthedocs.io/en/latest/encoders.html
#For example you can use "resnet18" for encoder and "imagenet" for weights
#You can choose one of them for activation “sigmoid”, “softmax”, “logsoftmax”, “tanh”, “identity”,

# Move model and loss to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = smp.losses.JaccardLoss(mode = "binary")
criterion.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0 
    total_iou_score = 0.0

    for step, (images, masks) in enumerate(train_loader, 1):
        images, masks = images.to(device), masks.to(device)
        # Forward pass
        outputs = model(images)

        # Calculate the loss
        loss = criterion(outputs, masks)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        tp, fp, fn, tn = smp.metrics.get_stats(outputs, masks, mode='binary', threshold=0.5)
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        total_iou_score += iou_score

        if step % print_interval == 0:
            
            average_iou_score = total_iou_score/len(train_loader)
            print(f"Epoch [{epoch + 1}/{epochs}], Step [{step}/{len(train_loader)}], Loss: {loss.item():.4f}, IoU: {iou_score:.4f},Average Iou Score:{average_iou_score:.4f}")

    average_loss = running_loss / len(train_loader)
    average_iou_score = total_iou_score / len(train_loader)
    print(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {average_loss:.4f},Training IoU: {average_iou_score:.4f}")

    # Validation loop
    model.eval()
    val_loss = 0.0
    val_correct_predictions = 0
    val_total_iou_score = 0.0

    with torch.no_grad():
        for val_images, val_masks in val_loader:
            val_images, val_masks = val_images.to(device), val_masks.to(device)
            # Forward pass
            val_outputs = model(val_images)
            # Calculate the loss
            val_loss = criterion(val_outputs, val_masks)
            val_loss += val_loss.item()
            tp, fp, fn, tn = smp.metrics.get_stats(val_outputs, val_masks, mode='binary', threshold=0.5)
            val_iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            val_total_iou_score += val_iou_score


    average_val_loss = val_loss / len(val_loader)
    average_val_iou_score = val_total_iou_score / len(val_loader)
    print(f"Epoch [{epoch + 1}/{epochs}], Validation Loss: {average_val_loss:.4f},  Validation IoU: {average_val_iou_score:.4f}")


# Testing loop
model.eval()
test_loss = 0.0
total_iou_score = 0.0

with torch.no_grad():
    for images, masks in test_loader:
        images, masks = images.to(device), masks.to(device)
        # Forward pass
        outputs = model(images)
        # Calculate the loss
        loss = criterion(outputs, masks)
        test_loss += loss.item()
        tp, fp, fn, tn = smp.metrics.get_stats(outputs, masks, mode='binary', threshold=0.5)
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        total_iou_score += iou_score
        
        if step % print_interval == 0:
            # then compute metrics with required reduction (see metric docs)
            average_iou_score = total_iou_score/len(test_loader)
            print(f"Epoch [{epoch + 1}/{epochs}], Step [{step}/{len(test_loader)}], Loss: {test_loss.item():.4f}, IoU: {iou_score:.4f},Average Iou Score:{average_iou_score:.4f}")

    average_loss = test_loss / len(test_loader)
    average_iou_score = total_iou_score / len(test_loader)
    print(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {average_loss:.4f},Training IoU: {average_iou_score:.4f}")
