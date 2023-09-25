from torch.utils.data import Dataset
import random
import os
from PIL import Image
import torch
import constants
import torch.nn.functional as F
import numpy as np


# 3. DataLoader Preparation
class VideoActionDataset(Dataset):
    def __init__(self, data, root_dir, transform=None):
        self.data = data
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def generate_random_bbox(self, width, height):
        """ Generate a random bounding box """
        xmin = random.randint(0, width - 100)  # assuming minimum width of bbox is 100
        ymin = random.randint(0, height - 100)  # assuming minimum height of bbox is 100
        xmax = random.randint(xmin + 100, width)
        ymax = random.randint(ymin + 100, height)
        return [xmin, ymin, xmax, ymax]
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.root_dir, item['filename'])
        image = Image.open(image_path).convert("RGB")
        
        annotations = item.get('annotations', [])
        
        # If there are no annotations, generate a random bounding box and assign label as background
        if not annotations:
            width, height = image.size
            random_bbox = self.generate_random_bbox(width, height)
            annotations = [{'name': '__background__', 'bbox': random_bbox}]
        
        # Convert category names to IDs and gather boxes
        boxes = []
        category_ids = []
        for annotation in annotations:
            boxes.append(annotation['bbox'])
            category_name = annotation.get('name')
            category_ids.append(constants.CATEGORY_MAPPING.get(category_name, 0))
        
        # Convert to numpy arrays
        image_np = np.array(image)
        boxes_np = np.array(boxes)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image_np, bboxes=boxes_np, category_ids=category_ids)
            image_np = transformed["image"]
            boxes_np = transformed["bboxes"]
        
        boxes = torch.as_tensor(boxes_np, dtype=torch.float32)

        return image_np, {"boxes": boxes, "labels": torch.tensor(category_ids, dtype=torch.int64)}

    
def collate_fn(batch):
    images = [item[0] for item in batch]
    annotations = [item[1] for item in batch]

    # Stack images, but keep annotations as a list of lists
    return torch.stack(images, 0), annotations
