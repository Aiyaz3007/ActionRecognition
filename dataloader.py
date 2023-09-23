from torch.utils.data import Dataset
import random
import os
from PIL import Image
import torch
import constants

# 3. DataLoader Preparation
class VideoActionDataset(Dataset):
    def __init__(self, data,root_dir, transform=None):
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
        width, height = image.size
        # Assuming you have loaded the image at this step (omitted for brevity)
        if self.transform:
            image = self.transform(image)

        annotations = item.get('annotations', [])
        if not annotations:  # If annotations list is empty
            random_bbox = self.generate_random_bbox(width, height)
            annotations.append({'bbox': random_bbox, 'category_id': 0})  # -1 for background class

        for annotation in annotations:
            category_name = annotation.get('name')
            if category_name:
                annotation['category_id'] = constants.CATEGORY_MAPPING.get(category_name, 0)
                del annotation['name']

        return image, annotations
    
def collate_fn(batch):
    images = [item[0] for item in batch]
    annotations = [item[1] for item in batch]

    # Stack images, but keep annotations as a list of lists
    return torch.stack(images, 0), annotations