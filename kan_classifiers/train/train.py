from ..models.sota.vit import CustomViT
from ..data.datapipeline import DataPipeline


def train(params):

    # EXPERIMENT
    exp_name = params['experiment']['exp_name']

    # MODEL
    model_name = params['model']['model_name']

    # OPTIMIZATION
    epoch = params['optimization']['epoch']
    lr = params['optimization']['lr']

    # DATASET
    dataset_name = params['dataset']['dataset_name']
    datapath = params['dataset']['datapath']
    batch_size = params['dataset']['batch_size']
    val_split = params['dataset']['val_split']
    num_workers = params['dataset']['num_workers']
    pin_memory = params['dataset']['pin_memory']

    # LOGGING
    log_folder = params['logging']['folder']
    
    models_factory = ModelsFactory()
    datapipeline = DataPipeline(datapath, batch_size, num_workers, pin_memory, val_split)

    model, processor = models_factory.load_model(model_name=model_name)





