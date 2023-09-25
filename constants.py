import torch

annotations_dir = '/content/dataset/annotations'
root_dir = "/content/dataset/images"

CATEGORY_MAPPING = {
    '__background__': 0,  
    'fighting': 1,
    'walking': 2,  
    'running': 3,  
    'sitting': 4,  
    'standing': 5,  
    'fallen': 6,  
    'accident': 7,  
}

num_classes = len(CATEGORY_MAPPING)

batch_size = 16

epochs = 10

loss_file = "loss.json"

model_saved_path = "logs"

using_notebook = True

sanity_check = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
