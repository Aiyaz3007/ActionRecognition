annotations_dir = 'dataset_video/Annotations'
root_dir = "dataset_video/JPEGImages"

CATEGORY_MAPPING = {
    '__background__': 0,  
    'fighting': 1,  
}

num_classes = len(CATEGORY_MAPPING)

batch_size = 8

epochs = 10

loss_file = "loss.json"

model_saved_path = "logs"


