from torchvision import transforms
from dataloader import VideoActionDataset,collate_fn
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import parse_xml,split_dataset,updateGraph
import constants
import os
import torch.nn as nn
import torchvision.models as models
from models import RNNModel
import torch
from tqdm import tqdm as bar
import json
from os.path import join


loss_format = {
    "train_loss":[],
    "val_loss":[]
}

# create loss json file 
with open(constants.loss_file,"w") as f:
    json.dump(loss_format,f)  

os.makedirs(join(constants.model_saved_path,"ssd"),exist_ok=True)
os.makedirs(join(constants.model_saved_path,"rnn"),exist_ok=True)



# Data Augmentation
transform = transforms.Compose([
    transforms.RandomCrop((500, 500)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    transforms.ToTensor()  # Convert to tensor for model input
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# getting parsed data from annotation 
parsed_data = [parse_xml(os.path.join(constants.annotations_dir, xml_file)) for xml_file in os.listdir(constants.annotations_dir)]

train_data, val_data, test_data = split_dataset(parsed_data)

train_dataset = VideoActionDataset(train_data,constants.root_dir, transform=transform)
val_dataset = VideoActionDataset(val_data, constants.root_dir,transform=transform)
test_dataset = VideoActionDataset(test_data, constants.root_dir,transform=transform)


# Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=constants.batch_size, shuffle=False,collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=constants.batch_size, shuffle=False,collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=constants.batch_size, shuffle=False,collate_fn=collate_fn)


ssd_model = models.detection.ssdlite320_mobilenet_v3_large(num_classes=constants.num_classes).to(device)
rnn_model = RNNModel(input_size=128, hidden_size=512, num_layers=2, num_classes=constants.num_classes).to(device)


# Define the loss functions
classification_loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(ssd_model.parameters()) + list(rnn_model.parameters()), lr=0.001)

epochBar = bar(total=constants.epochs,desc="Epoch")
trainBar = bar(total=len(train_loader),desc="Train")
valBar = bar(total=len(val_loader),desc="Val")



for epoch in range(constants.epochs):
    # Training phase
    ssd_model.train()
    rnn_model.train()
    train_total_loss = 0.0
    for batch_idx, (images, annotations) in enumerate(train_loader):
        images = images.to(device)
        
        # Extract actions and bboxes from annotations
        bboxes = []
        actions = []
        for annotation in annotations:
            if annotation:  # Check if annotation list is not empty
                for item in annotation:
                    bboxes.append(torch.tensor(item['bbox']))
                    actions.append(item['category_id'])

        if not bboxes:  # If no bounding boxes in this batch, skip this iteration
            continue

        bboxes = torch.stack(bboxes).float().to(device)  # Convert list of tensors to a single tensor
        actions = torch.tensor(actions).long().to(device)

        # Forward pass: Get SSD model output
        loss_dict = ssd_model(images, [{"boxes": bboxes, "labels": actions} for _ in range(images.size(0))])
        ssd_losses = sum(loss for loss in loss_dict.values())

        # Extract feature tensor from SSD for RNN input
        ssd_features = ssd_model.backbone(images)
        feature_tensor = ssd_features['5'].mean([2, 3])  # Global Average Pooling (GAP)

        # Forward pass: RNN
        rnn_output = rnn_model(feature_tensor)
        rnn_loss = classification_loss_fn(rnn_output, actions)
        
        # Compute the total loss and optimize
        train_loss = ssd_losses + rnn_loss
        train_total_loss += train_loss.item()
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        trainBar.update(1)
    
    train_total_loss = train_total_loss/len(train_loader)
    updateGraph(loss_type="train_loss",data=float(train_total_loss))
    

    # validation 
    ssd_model.train()
    rnn_model.eval()
    val_total_loss = 0.0
    for batch_idx, (images, annotations) in enumerate(val_loader):
        images = images.to(device)

        # Extract actions and bboxes from annotations
        bboxes = []
        actions = []
        for annotation in annotations:
            if annotation:  # Check if annotation list is not empty
                for item in annotation:
                    bboxes.append(torch.tensor(item['bbox']))
                    actions.append(item['category_id'])

        if not bboxes:  # If no bounding boxes in this batch, skip this iteration
            continue

        bboxes = torch.stack(bboxes).float().to(device)  # Convert list of tensors to a single tensor
        actions = torch.tensor(actions).long().to(device)

        # Forward pass: Get SSD model output
        with torch.no_grad():
            loss_dict = ssd_model(images, [{"boxes": bboxes, "labels": actions} for _ in range(images.size(0))])
        ssd_losses = sum(loss for loss in loss_dict.values())

        # Extract feature tensor from SSD for RNN input
        ssd_features = ssd_model.backbone(images)
        feature_tensor = ssd_features['5'].mean([2, 3])  # Global Average Pooling (GAP)

        # Forward pass: RNN
        rnn_output = rnn_model(feature_tensor)
        rnn_loss = classification_loss_fn(rnn_output, actions)
        val_loss = ssd_losses + rnn_loss
        val_total_loss += val_loss.item()
        valBar.update(1)

    ssd_model.eval()

    val_total_loss = val_total_loss/len(val_loader)
    updateGraph(loss_type="val_loss",data=float(val_total_loss))

    torch.save(ssd_model.state_dict(), os.path.join(constants.model_saved_path,"ssd", f'ssd_model_epoch_{epoch+1}_val_{round(val_total_loss,2)}.pth'))
    torch.save(rnn_model.state_dict(), os.path.join(constants.model_saved_path,"rnn", f'rnn_model_epoch_{epoch+1}_val_{round(val_total_loss,2)}.pth'))
    



    epochBar.update(1)
