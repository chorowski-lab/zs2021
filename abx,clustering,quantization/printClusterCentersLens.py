

import torch
import sys
import numpy as np

clustersFile = sys.argv[1]
clustersFileExt = clustersFile.split('.')[-1]
assert clustersFileExt in ('pt', 'npy', 'txt')
if clustersFileExt == 'npy':
    centers = torch.tensor(np.load(clustersFile), dtype=float)
elif clustersFileExt == 'txt':
    centers = torch.tensor(np.genfromtxt(clustersFile), dtype=float)
elif clustersFileExt == 'pt':
    centers = torch.load(clustersFile)['state_dict']['Ck']
    centers = torch.reshape(centers, centers.shape[1:])
    centers = torch.tensor(centers, dtype=float)

centersLengths = torch.sqrt((centers*centers).sum(1))
print(centersLengths)
print(str([(centers[i].min().item(), centers[i].max().item()) for i in range(centers.shape[0])]))
#print(centers[1])