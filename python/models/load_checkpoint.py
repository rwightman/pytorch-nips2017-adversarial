import torch
import os


def load_checkpoint(model, checkpoint_path):
    if checkpoint_path is not None and os.path.isfile(checkpoint_path):
        print('Loading checkpoint', checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("Error: No checkpoint found at %s." % checkpoint_path)
