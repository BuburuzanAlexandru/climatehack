import os
import numpy as np
import yaml
import torch
import matplotlib.pyplot as plt

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
    
    
def plot_preds(x, y, p):
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 12, figsize=(10,4))
    _min = np.min([x.min(), y.min(), p.min()])
    _max = np.max([x.max(), y.max(), p.max()])
    # plot the twelve 128x128 input images
    for i, img in enumerate(x):
        ax1[i].imshow(img, cmap='viridis', vmin=_min, vmax=_max)
        ax1[i].get_xaxis().set_visible(False)
        ax1[i].get_yaxis().set_visible(False)

    # plot twelve 64x64 true output images
    for i, img in enumerate(y[:12]):
        ax2[i].imshow(img, cmap='viridis', vmin=_min, vmax=_max)
        ax2[i].get_xaxis().set_visible(False)
        ax2[i].get_yaxis().set_visible(False)

    # plot twelve more 64x64 true output images
    for i, img in enumerate(y[12:]):
        ax3[i].imshow(img, cmap='viridis', vmin=_min, vmax=_max)
        ax3[i].get_xaxis().set_visible(False)
        ax3[i].get_yaxis().set_visible(False)

    # plot the twelve 64x64 predicted output images
    for i, img in enumerate(p[:12]):
        ax4[i].imshow(img, cmap='viridis', vmin=_min, vmax=_max)
        ax4[i].get_xaxis().set_visible(False)
        ax4[i].get_yaxis().set_visible(False)

    # plot twelve more 64x64 output images
    for i, img in enumerate(p[12:]):
        ax5[i].imshow(img, cmap='viridis', vmin=_min, vmax=_max)
        ax5[i].get_xaxis().set_visible(False)
        ax5[i].get_yaxis().set_visible(False)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    