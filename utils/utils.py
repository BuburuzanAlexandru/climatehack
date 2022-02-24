import os
import numpy as np
import yaml
import torch

def create_files(path_list):
    """Creates non-existing files"""

    for path in path_list:
        if os.path.exists(path) is False:
            os.mkdir(path)


def save_logs(logs, path):
    """Saves the config dict in yaml format"""

    with open(path + '/{}.yaml'.format('config_logs'), 'w') as output_file:
        yaml.dump(logs, output_file)


def get_lr(optimizer):
    """Returns current learning rate of optimizer"""

    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_model(epoch, model, optimizer, path):
    """Saves the state dict of the current model"""

    state_dict = model.state_dict()
    
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()
        
    torch.save({
        'epoch': epoch,
        'state_dict': state_dict,
        'optimizer' : optimizer.state_dict()},
        path + '/{}.pt'.format(epoch))