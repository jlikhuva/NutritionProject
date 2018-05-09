# Mostly Stolen From CS230.
# `https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/utils.py`
# Utility Functions for logging and Saving model.
import os
import torch
'''
 ##################
 # Example Usage  #
 ##################
state = {
    'epoch': epoch + 1,
    'state_dict': model.state_dict(),
    'optim_dict' : optimizer.state_dict()
}
utils.save_checkpoint(
    state,
    checkpoint=model_dir
)
'''

def save_checkpoint(state, checkpoint='../Data/FullData'):
    """
    Saves the best model and training parameters at
    `checkpoint + 'best_model.tar'``.
    Args:
        state:
            (dict) contains model's state_dict,
            may contain other keys such as epoch, optimizer state_dict
        checkpoint:
            (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'best_model.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    torch.save(state, filepath)


def load_checkpoint(checkpoint, model, optimizer=None):
    """
    Loads model parameters (state_dict) from file_path.
    If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint:
            (string) filename which needs to be loaded
        model:
            (torch.nn.Module) model for which the parameters are loaded
        optimizer:
            (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])
