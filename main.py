from torchvision import transforms
from dataloader import VideoActionDataset,collate_fn
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import parse_xml,split_dataset,updateGraph,sanity_check
import constants
import os
import torch.nn as nn
import torchvision.models as models
from models import RNNModel
import torch
import json
from os.path import join
import albumentations as A
from albumentations.pytorch import ToTensorV2

if constants.using_notebook:
  from tqdm.notebook import tqdm
else:
  from tqdm import tqdm

loss_format = {
    "train_loss":[],
    "val_loss":[]
}

# create loss json file 
with open(constants.loss_file,"w") as f:
    json.dump(loss_format,f)  

os.makedirs(join(constants.model_saved_path,"ssd"),exist_ok=True)
# os.makedirs(join(constants.model_saved_path,"rnn"),exist_ok=True)



transform = A.Compose(
    [
        A.Resize(300, 300),  # Resize both image and bounding boxes
        A.RandomCrop(256, 256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize images
        ToTensorV2()
    ],
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'])
)


device = constants.device

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

epochBar = tqdm(total=constants.epochs,desc="Epoch")
trainBar = tqdm(total=len(train_loader),desc="Train")
valBar = tqdm(total=len(val_loader),desc="Val")

if constants.sanity_check:
    sanity_check(dataset=train_dataset)

for epoch in range(constants.epochs):
    # Training phase
    ssd_model.train()
    train_total_loss = 0.0
    for batch_idx, (images, annotations) in enumerate(train_loader):
        images = images.to(device)
        loss_dict = ssd_model(images,annotations)
        ssd_losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        ssd_losses.backward()
        optimizer.step()
        train_total_loss += ssd_losses.item()
        trainBar.update(1)
    
    train_total_loss = train_total_loss/len(train_loader)
    updateGraph(loss_type="train_loss",data=float(train_total_loss))
    

    # validation 
    ssd_model.train()
    val_total_loss = 0.0
    for batch_idx, (images, annotations) in enumerate(val_loader):
        images = images.to(device)
        with torch.no_grad():
            loss_dict = ssd_model(images,annotations)
        ssd_losses = sum(loss for loss in loss_dict.values())
        val_total_loss += ssd_losses.item()
        valBar.update(1)

    val_total_loss = val_total_loss/len(val_loader)
    updateGraph(loss_type="val_loss",data=float(val_total_loss))

    torch.save(ssd_model.state_dict(), os.path.join(constants.model_saved_path,"ssd", f'ssd_model_epoch_{epoch+1}_val_{round(val_total_loss,2)}.pth'))
    epochBar.update(1)
