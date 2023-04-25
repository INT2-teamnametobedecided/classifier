#!/usr/bin/env python3

import torch
import torchvision.datasets.flowers102 as flowers102
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import scipy

dataset = flowers102.Flowers102(root="data", download=True)

labels = scipy.io.loadmat('data/flowers-102/imagelabels.mat')['labels']
labels_data = torch.from_numpy(labels).float()
print(f'labels: {labels_data}')

setid = scipy.io.loadmat('data/flowers-102/setid.mat')
#labels_data = torch.from_numpy(labels).float()
print(f'setid: {setid}')