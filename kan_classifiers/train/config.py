
class Params:
    DATASET_NAME = "cifar10"  # ["cifar10", "tiny_imagenet", "cifar100"]
    num_workers = 6
    batch_size = 64
    valid_size = 0.2
    MODEL_NAME = "ConvNet" #  ['FFNN', 'ConvNet', 'RESNET', 'ViT', 'CNN_KNN']
    num_workers = 4
    n_epochs = 50

    configs = {
        'cifar10': {'data_path':'/app/kan_classifiers/kan_classifiers/data/cifar10'},
        'cifar100': {'data_path':'/app/kan_classifiers/kan_classifiers/data/cifar100'},
        'tiny_imagenet': {'data_path':'/app/kan_classifiers/kan_classifiers/data/tiny_imagenet'}
    }

    model_path = "model.pt" # Absolute model path