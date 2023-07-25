import numpy as np
from torch.utils.data import Dataset
import torch
import random

class DataLoader():
    def __init__(self, training_params, dataset_params, tag=None, shuffle=None):
        self.ds = dataset_params
        self.tp = training_params
        # Reset a variable which will be recomputed but can cause problems down the line with subset sizes
        self.dataset= Dataset(self.ds['data_path'], self.tp['per_class'], self.tp['sample'], self.tp['device'])
        # shuffle = False: We shuffle data our own way
        if tag=='target' and self.tp['per_class']>0:
            s = Sampler(self.dataset.idx_supervision, self.tp['threshold'], len(self.dataset), self.ds['batch_size'], self.tp['device'])
            self.loader = torch.utils.data.DataLoader(self.dataset, batch_sampler=s)
        else:
            if shuffle is None:
                self.loader = torch.utils.data.DataLoader(self.dataset,
                                                          batch_size=self.ds['batch_size'], shuffle=self.ds['shuffle'],
                                                          drop_last=self.ds['drop_last'])
            else:
                self.loader = torch.utils.data.DataLoader(self.dataset,
                                                          batch_size=self.ds['batch_size'], shuffle=shuffle,
                                                          drop_last=self.ds['drop_last'])

    def get_loader(self):
        return self.loader, self.dataset

class Dataset(Dataset):
    def __init__(self, dataset_dir, nb_per_class, sample, device):
        self.dataset_dir = dataset_dir
        flag = True

        self.features = torch.from_numpy(np.load(dataset_dir + 'features/features.npy').astype(np.float32)).to(device)
        self.labels = torch.from_numpy(np.load(dataset_dir + 'labels/labels.npy').astype(np.float32)).to(device)

        self.idx_supervision = None
        classC = []
        labelsC = []

        if nb_per_class + sample > 0 and flag:

            if self.labels.ndim>1:
                flatLabels = torch.argmax(self.labels,-1)
            else:
                flatLabels=self.labels
            idx = []
            labelsU = torch.unique(flatLabels)

            for idC, c in enumerate(labelsU):
                idxClass = torch.argwhere(flatLabels == c).ravel()
                idxPerm = idxClass[torch.randperm(len(idxClass))]
                if sample > 0:
                    idxPermC = idxPerm[0:min(sample, len(idxClass))]
                    classC.append(self.features[idxPermC])
                    labelsC.append(self.labels[idxPermC])
                    idxClassSampled = torch.randperm(len(classC[-1]))[:nb_per_class]+ sample*idC
                else:
                    idxClassSampled = idxPerm[0:min(nb_per_class, len(idxClass))]
                idx.extend(idxClassSampled)
            if nb_per_class>0:
                self.idx_supervision = torch.stack(idx).to(device)

        if len(classC)>0:
            self.features = torch.cat(classC).to(device)
            self.labels = torch.cat(labelsC).to(device)

        if flag:
            assert self.features.shape[0] == self.labels.shape[0]

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        if type(self.labels) is not dict and self.labels != []:
            return self.features[idx], self.labels[idx], idx
        else:
            return self.features[idx], idx


class Sampler():
    def __init__(self, idx_supervision, threshold, size, batch_size, device):
        self.idx_supervision = idx_supervision
        self.n_batches = size // batch_size
        self.size = size
        self.batch_size = batch_size
        self.threshold = threshold
        self.device = device

    def __iter__(self):
        batches = []
        threshold = int(self.threshold*self.batch_size)
        idx_unsupervised = [torch.tensor(idx) for idx in list(range(self.size)) if idx not in self.idx_supervision]

        if len(self.idx_supervision)>threshold:
            permIdx = self.idx_supervision[torch.randperm(len(self.idx_supervision))]
            toFreeze = permIdx[:threshold]
            toSample = self.batch_size-len(toFreeze)
        else:
            toSample = self.batch_size-len(self.idx_supervision)
            toFreeze = self.idx_supervision
        for _ in range(self.n_batches):
            indices = [idx_unsupervised[i].to(self.device) for i in torch.randperm(len(idx_unsupervised))[:toSample]]
            batch = [el.to(self.device) for el in toFreeze] + indices
            batches.append(batch)
        return iter(batches)

    def __len__(self):
        return self.batch_size
