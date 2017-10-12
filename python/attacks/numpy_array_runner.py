import torch.utils.data as data
import numpy as np

class NumpyArrayAttackRunner:
    def __init__(self, dataset):
        self.dataset = dataset

    def run(self, attack, batch_size):
        loader = data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False)

        output = []
        for input, target in loader:
            input, target = data.cuda(), target.cuda()
            input, _, _ = attack(input, target, None)
            output.append(input.cpu().numpy().squeeze())
            print('Completed so far: {}'.format(sum([len(x) for x in output])))

        return np.concatenate(output)