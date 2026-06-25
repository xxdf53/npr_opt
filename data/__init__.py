import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

from .datasets import dataset_folder

'''
def get_dataset(opt):
    dset_lst = []
    for cls in opt.classes:
        root = opt.dataroot + '/' + cls
        dset = dataset_folder(opt, root)
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst)
'''

import os
def _load_one_dataset(opt, dataroot):
    """Load a single dataset from dataroot (handles both flat 0_real/1_fake
    and nested multi-generator structures)."""
    classes = os.listdir(dataroot) if len(opt.classes) == 0 else opt.classes
    if '0_real' not in classes or '1_fake' not in classes:
        # Nested: each subdir is a generator with its own 0_real/1_fake
        dset_lst = []
        for cls in sorted(classes):
            root = os.path.join(dataroot, cls)
            if os.path.isdir(root):
                dset_lst.append(dataset_folder(opt, root))
        return torch.utils.data.ConcatDataset(dset_lst)
    return dataset_folder(opt, dataroot)

def get_dataset(opt):
    dataset = _load_one_dataset(opt, opt.dataroot)
    extra = getattr(opt, 'extra_dataroot', None)
    if extra:
        print(f"Mixing extra dataset: {extra}")
        extra_dset = _load_one_dataset(opt, extra)
        dataset = torch.utils.data.ConcatDataset([dataset, extra_dset])
    return dataset

def _collect_targets(dataset):
    """Recursively collect targets from ImageFolder or ConcatDataset."""
    if hasattr(dataset, 'targets'):
        return list(dataset.targets)
    if hasattr(dataset, 'datasets'):
        targets = []
        for d in dataset.datasets:
            targets.extend(_collect_targets(d))
        return targets
    raise AttributeError(f'Dataset {type(dataset).__name__} has no targets or datasets')


def get_bal_sampler(dataset):
    targets = _collect_targets(dataset)

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler


def create_dataloader(opt):
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    dataset = get_dataset(opt)
    sampler = get_bal_sampler(dataset) if opt.class_bal else None

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              sampler=sampler,
                                              num_workers=int(opt.num_threads))
    return data_loader
