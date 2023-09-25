annotations_dir = 'result/annotations'
root_dir = "result/images"

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

batch_size = 8

epochs = 10

loss_file = "loss.json"

model_saved_path = "logs"

sanity_check = False
