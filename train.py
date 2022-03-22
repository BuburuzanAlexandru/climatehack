import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.dataset_preprocessed import ClimatehackDatasetPreprocessed
from utils.loss import MS_SSIMLoss
from utils.utils import *
from config import config, results_config
from tqdm import tqdm

from submission.ConvLSTM2 import ConvLSTM

device = torch.device(config['device'])
create_files(results_config.values())
save_logs(config, config['logs_path'])
writer = SummaryWriter(config['tensorboard_path'])

def train_one_epoch(model, optimizer, train_dataloader, writer, criterion, epoch):
    """
    Train
    """
    train_loss = 0
    count = 0
    model.train()

    for inputs, target in tqdm(train_dataloader):
        inputs = inputs.to(device)
        target = target.to(device)
        
        # clear gradients
        optimizer.zero_grad()
        
        # forward
        output = model(inputs)
        batch_loss = criterion(output, target)
        
        # compute gradients
        batch_loss.backward()
        
        # update weights
        optimizer.step()
        
        # update stats
        train_loss += batch_loss.item() * output.shape[0]
        count += output.shape[0]
        
        # scheduler.step()

        del inputs, target, output, batch_loss
    
    print('Train loss:', train_loss / count, epoch)
    writer.add_scalar('Train/loss', train_loss / count, epoch)

def valid(model, optimizer, valid_dataloader, writer, criterion, epoch):
    """
    Valid
    """
    valid_loss = 0
    count = 0
    model.eval()

    with torch.no_grad():
        for inputs, target in tqdm(valid_dataloader):
            inputs = inputs.to(device)
            target = target.to(device)
            
            output = model(inputs)
            batch_loss = criterion(output, target)
            
            valid_loss += batch_loss.item() * output.shape[0]
            count += output.shape[0]
            
            del inputs, target, output, batch_loss
    
    print('Valid loss:', valid_loss / count, epoch)
    writer.add_scalar('Valid/loss', valid_loss / count, epoch)
    
    save_model(epoch, model, optimizer, config['checkpoints_path'])


def main():
    train_dataset = ClimatehackDatasetPreprocessed(os.path.join(config['data_path'], 'train'),
                                                   os.path.join(config['data_path'], 'train_list.json'))
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=True)

    valid_dataset = ClimatehackDatasetPreprocessed(os.path.join(config['data_path'], 'valid'),
                                                   os.path.join(config['data_path'], 'valid_list.json'))
    valid_dataloader = DataLoader(valid_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=False)

    model = ConvLSTM()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    criterion = MS_SSIMLoss(channels=24) # produces less blurry images than nn.MSELoss()

    print('Successfully created model')

    for epoch in range(config['num_epochs']):
        print('Epoch', epoch)
        train_one_epoch(model, optimizer, train_dataloader, writer, criterion, epoch)
        valid(model, optimizer, valid_dataloader, writer, criterion, epoch)
    

if __name__ == '__main__':
    main()