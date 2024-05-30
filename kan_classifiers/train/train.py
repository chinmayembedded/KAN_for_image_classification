from ..models.sota.vit import CustomViT
from ..data.datapipeline import DataPipeline
from tqdm.auto import tqdm
import torch.optim as optim
import torch, os
from kan_classifiers.utils.logging import get_logger, CSVLogger
import torch.nn as nn
import time


def load_checkpoint(parent_folder, exp_name):
    full_path = os.path.join(parent_folder, exp_name)
    if os.path.exists(full_path):
        # load checkpoint
        pass
    else:
        return None
    

def load_optimizer(optim_name, model, lr):

    if optim_name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr)
    else:
        raise Exception(f'{optim_name} not available!')


def load_criterion(criterion_name):
    if criterion_name.lower() == 'crossentropy':
        return nn.CrossEntropyLoss()
    else:
        raise Exception(f'{criterion_name} not found!')
    

# Function to format elapsed time in hh:mm:ss
def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{int(hours):02}:{int(minutes):02}:{int(seconds):02}'


def train(params):

    # EXPERIMENT
    exp_name = params['experiment']['exp_name']
    exp_parent_path = params['experiment']['exp_parent_folder']
    output_dir = params['experiment']['output_dir']
    save_every = params['experiment']['save_every']

    # MODEL
    model_name = params['model']['model_name']
    hidden_features = params['model']['hidden_features']

    # OPTIMIZATION
    n_epochs = params['optimization']['epoch']
    lr = float(params['optimization']['lr'])
    device = params['optimization']['device']
    optim_name = params['optimization']['optim_name']
    criterion = params['optimization']['criterion']

    # DATASET
    dataset_name = params['dataset']['dataset_name']
    n_categories = params['dataset']['n_categories']
    datapath = params['dataset']['datapath']
    batch_size = params['dataset']['batch_size']
    val_split = params['dataset']['val_split']
    num_workers = params['dataset']['num_workers']
    pin_memory = params['dataset']['pin_memory']

    # LOGGING
    log_folder = params['logging']['folder']

    log_exp_path = os.path.join(log_folder, exp_name)
    
    train_logger = CSVLogger(os.path.join(log_exp_path, 'train_history.csv'), ['train_loss', 'val_loss', 'epoch', 'time'])
    info_logger = get_logger(filename=os.path.join(log_exp_path, 'info_train.log'), force=True)

    start_epoch = 0
    data_pipeline = DataPipeline(
        dataset_name=dataset_name,
        datapath=datapath, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        val_split=val_split
        )
    cifar10_train, cifar10_val, cifar10_test = data_pipeline.get_dataloader()
    model = CustomViT(out_features=n_categories, hidden_features=hidden_features).to(device)

    try:
        optimizer = load_optimizer(optim_name, model, lr)
    except Exception as e:
        info_logger.info(e.args[0], exc_info=True)
        exit()

    try:
        criterion = load_criterion(criterion)
    except Exception as e:
        info_logger.info(e.args[0], exc_info=True)
        exit()

    checkpoint = load_checkpoint(exp_parent_path, exp_name)
    if checkpoint:
        #load checkpoint here
        pass
    
    best_val_loss = float('inf')

    for e in range(start_epoch+1, n_epochs+1):
        info_logger.info(f'Start epoch {e}/{n_epochs}')
        start_time = time.time()
        # train loop
        train_loop = tqdm(cifar10_train, total=len(cifar10_train))
        running_train_loss = 0

        model.train()
        for data, labels in train_loop:
            optimizer.zero_grad()
            data = data.to(device)
            labels = labels.to(device)
            pred = model(data)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        
        epoch_train_loss = running_train_loss / len(cifar10_train)

        # val loop
        val_loop = tqdm(cifar10_val, total=len(cifar10_val))
        running_val_loss = 0
        model.eval()
        with torch.inference_mode():
            for data, labels in val_loop:
                data = data.to(device)
                labels = labels.to(device)
                pred = model(data)
                loss = criterion(pred, labels)
                running_val_loss += loss.item()
        
        epoch_val_loss = running_val_loss / len(cifar10_val)
        

        end_time = time.time()
        info_logger.info(f'Epoch {e+1}/{n_epochs} completed')
        train_logger.log({'train_loss':epoch_train_loss, 'val_loss':epoch_val_loss, 'epoch':e+1, 'time':format_time(end_time-start_time)})

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss

        if e % save_every == 0:
            info_logger.info(f'Saving checkpoint at epoch: {e+1}')
            checkpoint = {
                'epoch':e,
                'optimizer':optimizer.state_dict(),
                'model':model.state_dict()
            }
            torch.save(log_exp_path = os.path.join(log_folder, exp_name, 'checkpoint.pth'))
    
    info_logger.info(f'Training finished!')
    info_logger.info('Saving model...')
    torch.save(os.path.join(log_folder, exp_name, 'model.pth'))
    info_logger.info('Saving model...')














