

import torch
import os
import sys
from shutil import copyfile
import numpy as np
from dsgetter import *
import multiprocessing

def seDistancesToCentroidsCpy(vecs, centroids, doNorm=False):
    #print(torch.square(centroids).sum(1).view(1,-1).shape, torch.square(vecs).sum(1).view(-1,1).shape, torch.matmul(vecs, centroids.T).shape)
    if len(vecs.shape) == 2:
        vecs = vecs.view(1, *(vecs.shape))
    B = vecs.shape[0]
    N = vecs.shape[1]
    k = centroids.shape[0]
    # vecs: B x L x Dim
    # centroids: k x Dim
    if doNorm:
        vecLengths = torch.sqrt((vecs*vecs).sum(-1))
        vecs = vecs / vecLengths.view(B, N, 1)
        centrLengths = torch.sqrt((centroids*centroids).sum(-1))
        centroids = centroids / centrLengths.view(k, 1)
        # print(vecLengths)
        # print(centrLengths)
        # vecLengths2 = (vecs*vecs).sum(-1)
        # print(f'vec lengths after norm from {vecLengths2.min().item()} to {vecLengths2.max().item()}')
        # centrLengths2 = (centroids*centroids).sum(-1)
        # print(f'center lengths after norm from {centrLengths2.min().item()} to {centrLengths2.max().item()}')
    # print(torch.square(centroids).sum(1).view(1, 1, -1).shape, torch.square(vecs).sum(-1).view(B, N, 1).shape,
    #     (vecs.view(B, N, 1, -1) * centroids.view(1, 1, k, -1)).sum(-1).shape,
    #     vecs.view(B, N, 1, -1).shape, centroids.view(1, 1, k, -1).shape)
    return torch.square(centroids).sum(1).view(1, 1, -1) + torch.square(vecs).sum(-1).view(B, N, 1) \
        - 2*(vecs.view(B, N, 1, -1) * centroids.view(1, 1, k, -1)).sum(-1)  #torch.matmul(vecs, centroids.T)

def pushToClosestForLine(points, centers, deg=0.5, doNorm=False):

    distsSq = seDistancesToCentroidsCpy(points, centers, doNorm)
    dists = torch.sqrt(distsSq)
    dists = dists.view(*dists.shape[1:])
    k = dists.shape[1]
    N = points.shape[0]
    #print("!!!!", dists.shape)
    # print(dists.shape)
    closest = dists.argmin(1)
    # print(points.shape, closest.shape, centers[closest].view(N, -1).shape)
    diffs = centers[closest].view(N, -1) - points
    # print(diffs.shape)
    res = deg * diffs + points
    # print(res.shape)
    
    return res

def computeAndSaveForLine(ar):

    doCpy, lineHere, centersHere, degHere, nameHere, outHereH, doNorm = ar

    #for i in range(batch.shape[0]):
    if doCpy != 'yes':
        encoded2 = pushToClosestForLine(lineHere, centersHere, deg=degHere, doNorm=doNorm)
    else:
        encoded2 = lineHere
    #print(encoded2.shape)
    #print(outHere, names[i].split('.')[0] + ".txt")
    np.savetxt(os.path.join(outHereH, nameHere.split('.')[0] + ".txt"), np.array(encoded2))


def pushToClosestForBatch(points, centers, deg=0.5, doNorm=False):

    distsSq = seDistancesToCentroidsCpy(points, centers, doNorm=doNorm)
    dists = torch.sqrt(distsSq)
    k = dists.shape[1]
    N = points.shape[1]
    B = points.shape[0]
    
    # print(dists.shape)
    closest = dists.argmin(-1)
    # print(points.shape, closest.shape, centers[closest].view(B, N, -1).shape)
    diffs = centers[closest].view(B, N, -1) - points
    # print(diffs.shape)
    res = deg * diffs + points
    # print(res.shape)
    
    return res

def computeAndSaveForBatch(doCpy, batch, centersHere, degHere, namesHere, lengthsHere, outHereH):

    #doCpy, lineHere, centersHere, degHere, nameHere, outHereH = ar
    B = batch.shape[0]
    #for i in range(batch.shape[0]):
    if doCpy != 'yes':
        encoded2 = pushToClosestForBatch(batch, centersHere, deg=degHere)
    else:
        encoded2 = lineHere
    #print(encoded2.shape)
    #print(outHere, names[i].split('.')[0] + ".txt")
    for i in range(B):
        np.savetxt(os.path.join(outHereH, namesHere[i].split('.')[0] + ".txt"), np.array(encoded2[i][:lengthsHere[i]].cpu()))

if __name__ == '__main__':

    # line0 = torch.tensor([[1,1],[1,1],[2,2.1],[2,2.1],[2,2.1],[2,2.1],[3,3.3],[3,3.3],[3,3.3]], dtype=float)
    # line1 = torch.tensor([[1,1],[0.2,0.2],[2,2.1],[2,2.1],[2,2.1],[2,2.1],[3,3.3],[3,3.3],[3,3.3]], dtype=float)
    # lines = torch.zeros((2, *(line0.shape)), dtype=float)
    # lines[0] = line0
    # lines[1] = line1
    
    # centers0 = torch.tensor([[1.601,1.601],[2.601,2.701],[3.601,3.901]], dtype=float)

    # print(pushToClosestForLine(line0, centers0, deg=0.5, doNorm=True))  
    # print("lines: ", lines.shape)
    # print(pushToClosestForBatch(lines, centers0, deg=0.5, doNorm=True))  

    #DSgenPath = sys.argv[1]
    clustersFile = sys.argv[1]
    rootOutPathNoMPrefix = sys.argv[2]

    outSaveOpts = sys.argv[3].split(':')
    outSaveDirs = list(map(lambda x: x.split('{}'), outSaveOpts)) if outSaveOpts is not None else None  # here given arg is already split by :
    print(outSaveDirs)
    # if outSaveDirs is not None:
    #     for coding, p in outSaveDirs:
    #         assert p != ""
    #         if not os.path.exists(p):
    #             os.makedirs(p)

    metadataFileToCpy = sys.argv[4]
    filesInBatch = int(sys.argv[5])
    # shorteningNameSuffix = shortening.replace('.', '_')
    # shorteningRoot = rootOutPathPrefix + "_" + shorteningNameSuffix
    # #if not os.path.exists(shorteningRoot + "/phonetic"):
    # if outSaveFormat != 'nums':
    
    poolSettings = sys.argv[6]
    

    justCpy = sys.argv[7]
    degs = list(map(float, sys.argv[8].split(':')))
    print(degs)
    # weighted = sys.argv[8]
    # weighted = weighted == 'weighted'  
    # 
    doNorm = True if sys.argv[9] in ("True", "true") else False   # makes dists cosine  
    print(f"Norm: {doNorm}")    
    
    clustersFileExt = clustersFile.split('.')[-1]
    assert clustersFileExt in ('pt', 'npy', 'txt')
    if clustersFileExt == 'npy':
        centers = np.load(clustersFile)
    elif clustersFileExt == 'txt':
        centers = np.genfromtxt(clustersFile)
    elif clustersFileExt == 'pt':
        centers = torch.load(clustersFile, map_location=torch.device('cpu'))['state_dict']['Ck']
        centers = torch.reshape(centers, centers.shape[1:]).numpy()

    centers = torch.tensor(centers, dtype=float)

    if poolSettings != "cuda":
        pool = multiprocessing.Pool(int(poolSettings))
    else:
        centers = centers.cuda()

    for deg in degs:

        rootOutPathPrefix = rootOutPathNoMPrefix + "_deg" + str(deg).replace('.','-')
        print("---->", rootOutPathPrefix)

        if not os.path.exists(rootOutPathPrefix):
            os.makedirs(rootOutPathPrefix)
        os.makedirs(rootOutPathPrefix + "/phonetic")
        #if not os.path.exists(shorteningRoot + "/lexical"):
        os.makedirs(rootOutPathPrefix + "/lexical")
        #if not os.path.exists(shorteningRoot + "/syntactic"):
        os.makedirs(rootOutPathPrefix + "/syntactic")
        #if not os.path.exists(shorteningRoot + "/semantic"):
        os.makedirs(rootOutPathPrefix + "/semantic")
        #print(metadataFileToCpy)
        copyfile(metadataFileToCpy, rootOutPathPrefix + "/meta.yaml")

        #print(rootOutPathPrefix)
        savePathRoot = os.path.join(rootOutPathPrefix, 'phonetic')
        #print(savePathRoot)
        

        for outSuffix, subsetInPath in outSaveDirs:

            outHere = os.path.join(savePathRoot, outSuffix)
            if not os.path.exists(outHere):
                os.makedirs(outHere)

            print("loading DS")
            DSgen = DSgetter(subsetInPath, lambda x: np.genfromtxt(x), nameMustMatch=".*\.txt")

            batchGen = DSgen.batchGenerator(filesInBatch=filesInBatch)

            b = 1

            for lines, padMask, sumlen, lengths, names in batchGen:

                print(f'batch {b}')
                b += 1

                batch = torch.tensor(lines, dtype=float)

                if poolSettings != "cuda":

                    mapArgs = [(justCpy, batch[i][:lengths[i]], centers, deg, names[i], outHere, doNorm) for i in range(batch.shape[0])]

                    pool.map(computeAndSaveForLine, mapArgs)

                else:
                    batch = batch.cuda()
                    computeAndSaveForBatch(justCpy, batch, centers, deg, names, lengths, outHere, doNorm)


