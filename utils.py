import xml.etree.ElementTree as ET
import random
import constants
import json

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

