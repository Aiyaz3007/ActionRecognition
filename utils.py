import xml.etree.ElementTree as ET
import random
import constants
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import torch

# 1. Data Parsing
def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    depth_tag = size.find('depth')
    depth = int(depth_tag.text) if depth_tag is not None and depth_tag.text is not None else 3
    annotations = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = round(float(bndbox.find('xmin').text))
        ymin = round(float(bndbox.find('ymin').text))
        xmax = round(float(bndbox.find('xmax').text))
        ymax = round(float(bndbox.find('ymax').text))
        annotations.append({'name': name, 'bbox': [xmin, ymin, xmax, ymax]})
    return {'filename': filename, 'width': width, 'height': height, 'depth': depth, 'annotations': annotations}


# 2. Dataset Splitting
def split_dataset(data, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    assert (train_ratio + (val_ratio + test_ratio)) == 1.0, "Ratios must sum up to 1.0"
    random.shuffle(data)
    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    return train_data, val_data, test_data


def updateGraph(loss_type:str,data):
    with open(constants.loss_file,"r") as f:
        loss = json.load(f)
    assert loss_type in ["train_loss", "val_loss"], "loss type is different"
    if loss_type == "train_loss":
        loss["train_loss"].append(float(data))

    elif loss_type == "val_loss":
        loss["val_loss"].append(float(data))

    with open(constants.loss_file,"w") as f:
        json.dump(loss,f)

def denormalize(tensor, mean, std):
    """
    Denormalize a tensor.
    
    Parameters:
        tensor (torch.Tensor): The normalized tensor.
        mean (list or tuple): The mean used during normalization.
        std (list or tuple): The standard deviation used during normalization.
        
    Returns:
        torch.Tensor: The denormalized tensor.
    """
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(constants.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(constants.device)
    return tensor * std + mean

# # Example usage:
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
# normalized_tensor = ... # Assuming this is your normalized tensor
# denormalized_tensor = denormalize(normalized_tensor, mean, std)

def sanity_check(dataset):
    # Create a new directory for saving the sanity check images
    image_count = 5
    bar = tqdm(total=image_count,desc="sanity")
    output_folder = "sanity_samples"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    def save_sample(dataset, idx, output_path):
        image, annotation = dataset[idx]
        boxes = annotation['boxes'].numpy()
        labels = annotation['labels'].numpy()
        image = denormalize(tensor=image,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])[0]
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(image.permute(1, 2, 0))  # Convert CxHxW to HxWxC

        for box, label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(xmin, ymin, s=list(constants.CATEGORY_MAPPING.keys())[list(constants.CATEGORY_MAPPING.values()).index(label)], color='white', verticalalignment='top',
                    bbox={'color': 'red', 'pad': 0})
        
        plt.savefig(output_path)
        plt.close()

    # Randomly sample and save a few samples from the training dataset
    for i in range(5):
        idx = random.randint(0, len(dataset) - 1)
        output_path = os.path.join(output_folder, f"sample_{i}.jpg")
        save_sample(dataset, idx, output_path)
        bar.update(1)

    print(f"Saved sanity check images in '{output_folder}' folder.")
