import torch
import torchvision.datasets as datasets

class CIFAR10(datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        super(CIFAR10, self).__init__(*args, **kwargs)
        self.targets = torch.tensor(self.targets)

    def __getitem__(self, index):
        out_super = datasets.CIFAR10.__getitem__(self, index)
        return *out_super , index