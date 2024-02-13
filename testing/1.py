import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt("./testing/datasets/fam.csv", delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:,1:])
        self.y =  torch.from_numpy(xy[:,[0]])
        self.num_samples = xy.shape[0]
        
        super().__init__()

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.num_samples
    
    
dataset = WDataset()

dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

dataitter = iter(dataloader)

data = dataitter._next_data()
features, labels = data
totalSamples = len(dataset)
n_iterations =math.ceil(totalSamples/4)
for epoch in range(1000):

    for i, (inputs,label) in enumerate(dataloader):
        if (i+1)% 5 == 0:
            print(f'epoch {epoch+1} step {i+1}')
            

print(features, labels)