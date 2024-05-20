
class Params:
    DATASET_NAME = "CIFAR"  # CIFAR / MINIIMAGENET
    num_workers = 6
    batch_size = 256
    valid_size = 0.3
    MODEL_TYPE = "CNN" #  ['FFNN', 'CNN', 'RESNET', 'ViT', 'CNN_KNN']