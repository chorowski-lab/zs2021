
import sys
import os
import numpy as np
import torch

def seDistancesOfCenters(centers):
    #print(np.square(centers).sum(1)[:,np.newaxis].shape, np.square(centers).sum(1)[:,np.newaxis])
    #print(np.square(centers).sum(1)[np.newaxis,:].shape, np.square(centers).sum(1)[np.newaxis,:])
    #print(np.matmul(centers, centers.T).shape, np.matmul(centers, centers.T))
    return np.square(centers).sum(1)[:,np.newaxis] + np.square(centers).sum(1)[np.newaxis,:] - 2*np.matmul(centers, centers.T)  


clustersFile = sys.argv[1]
outDir = sys.argv[2]
doSq = sys.argv[3]
assert doSq in ('sq', 'lin', 'cosIfNormed', 'cosSqIfNormed')

clustersFileExt = clustersFile.split('.')[-1]
assert clustersFileExt in ('pt', 'npy', 'txt')
if clustersFileExt == 'npy':
    centers = np.load(clustersFile)
elif clustersFileExt == 'txt':
    centers = np.genfromtxt(clustersFile)
elif clustersFileExt == 'pt':
    centers = torch.load(clustersFile, map_location=torch.device('cpu'))['state_dict']['Ck']
    centers = torch.reshape(centers, centers.shape[1:]).numpy()

print(centers.shape)

if doSq == 'sq':
    dists = seDistancesOfCenters(centers)
elif doSq == 'lin':
    dists = np.sqrt(seDistancesOfCenters(centers))  #np.sqrt(seDistancesOfCenters(centers))
elif doSq == 'cosSqIfNormed':
    dists = np.square(seDistancesOfCenters(centers) / 2.)
elif doSq == 'cosIfNormed':
    dists = seDistancesOfCenters(centers) / 2.

dists = np.nan_to_num(np.maximum(dists, 0.))
print(dists)

print(dists.shape, dists.min(), dists.max())

if not os.path.exists(outDir):
   os.makedirs(outDir)

np.savetxt(os.path.join(outDir, "distMatrix.txt"), dists)
