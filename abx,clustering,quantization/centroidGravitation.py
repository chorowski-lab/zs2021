

import torch
import os
import sys
from shutil import copyfile
import numpy as np
from dsgetter import *
import multiprocessing


def seDistancesToCentroids(vecs, centroids, doNorm=False):
    
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
        
    return torch.square(centroids).sum(1).view(1, 1, -1) + torch.square(vecs).sum(-1).view(B, N, 1) \
        - 2*(vecs.view(B, N, 1, -1) * centroids.view(1, 1, k, -1)).sum(-1)  #torch.matmul(vecs, centroids.T)


def pushToClosestForLine(points, centers, deg=0.5, doNorm=False, doNormForPush=False):

    N = points.shape[0]
    k = centers.shape[0]

    if doNormForPush:
        pointsLengths = torch.sqrt((points*points).sum(-1))
        points = points / pointsLengths.view(N, 1)
        centrLengths = torch.sqrt((centers*centers).sum(-1))
        centers = centers / centrLengths.view(k, 1)
        
    distsSq = seDistancesToCentroids(points, centers, doNorm=doNorm)
    dists = torch.sqrt(distsSq)
    dists = dists.view(*dists.shape[1:])
    
    closest = dists.argmin(1)
    diffs = centers[closest].view(N, -1) - points
    res = deg * diffs + points
    
    return res


def computeAndSaveForLine(ar):

    line, centers, deg, name, out, doNorm, doNormForPush, isDebug = ar

    encoded2 = pushToClosestForLine(line, centers, deg=deg, doNorm=doNorm, doNormForPush=doNormForPush)
    
    if not isDebug:
        np.savetxt(os.path.join(out, name.split('.')[0] + ".txt"), np.array(encoded2))
    else:
        return encoded2


# actually now unused even for cuda variant which now glues batch into 1 line
def pushToClosestForBatch(points, centers, deg=0.5, doNorm=False, doNormForPush=False):

    B = points.shape[0]   
    N = points.shape[1]
    k = centers.shape[0]

    if doNormForPush:
        pointsLengths = torch.sqrt((points*points).sum(-1))
        points = points / pointsLengths.view(B, N, 1)
        centrLengths = torch.sqrt((centers*centers).sum(-1))
        centers = centers / centrLengths.view(k, 1)

    distsSq = seDistancesToCentroids(points, centers, doNorm=doNorm)
    dists = torch.sqrt(distsSq)
     
    closest = dists.argmin(-1)
    diffs = centers[closest].view(B, N, -1) - points
    res = deg * diffs + points
     
    return res


def computeAndSaveForBatch(batch, centers, deg, names, lengths, out, doNorm=False, doNormForPush=False, isDebug=False):

    B = batch.shape[0]

    # actually not needed to make a shape with B, averagedLen, Dim
    # can just cat and then restore for result
    #sumlen = sum(lengths)
    #linelen = int(float(sumlen + B - .99) // B)

    batch1line = torch.cat([batch[i] for i in range(B)], dim=0)

    encoded2 = pushToClosestForLine(batch1line, centers, deg=deg, doNorm=doNorm, doNormForPush=doNormForPush)
    
    if isDebug:
        result = torch.full(batch.shape, -1, dtype=float).cuda()
        
    currentBegin = 0
    for i in range(B):
        currentEnd = min(currentBegin + lengths[i], encoded2.shape[0])
        if not isDebug:
            np.savetxt(os.path.join(out, names[i].split('.')[0] + ".txt"), np.array(encoded2[currentBegin:currentEnd].cpu()))
        else:
            result[i][:(currentEnd-currentBegin)] = encoded2[currentBegin:currentEnd]
        currentBegin = currentEnd

    if isDebug:
        return result


if __name__ == '__main__':

    if sys.argv[1] == 'debug':

        line0 = torch.tensor([[1,1],[1,1],[2,2.1],[2,2.1],[2,2.1],[2,2.1],[3,3.3],[3,3.3],[3,3.3]], dtype=float)
        line1 = torch.tensor([[1,1],[0.2,0.2],[2,2.1],[2,2.1],[2,2.1],[2,2.1],[3,3.3],[3,3.3],[-1,-1]], dtype=float)
        lines = torch.zeros((2, *(line0.shape)), dtype=float)
        lines[0] = line0
        lines[1] = line1
        
        centers0 = torch.tensor([[1.601,1.601],[2.601,2.701],[3.601,3.901]], dtype=float)

        print(computeAndSaveForLine((line0, centers0, 0.5, None, None, True, False, True)))  
        print("lines: ", lines.shape)
        print(computeAndSaveForBatch(lines, centers0, 0.5, None, [9,8], None, doNorm=True, isDebug=True)) 
        sys.stdout.flush()

        exit() 

    clustersFile = sys.argv[1]
    rootOutPathNoMPrefix = sys.argv[2]

    outSaveOpts = sys.argv[3].split(':')
    outSaveDirs = list(map(lambda x: x.split('{}'), outSaveOpts)) if outSaveOpts is not None else None  # here given arg is already split by :
    print(outSaveDirs)
    
    metadataFileToCpy = sys.argv[4]
    filesInBatch = int(sys.argv[5])
    
    poolSettings = sys.argv[6]
    
    #justCpy = sys.argv[7]
    degs = list(map(lambda x: (float(x), x), sys.argv[7].split(':')))
    print(degs)
    
    # makes dists cosine
    doNorm = True if sys.argv[8] in ("True", "true") else False  
    
    # approximation of pushing %age of cosine and not euclid dist
    # the bigger the angle the worse approximation and less pushing
    # could also solve some equation for actual %age of angle, but this is possible future TODO
    # without doNormForPush pushing is euclidean and results in smaller-angle push for euclid-further points
    doNormForPush = True if sys.argv[9] in ("True", "true") else False
    print(f"Norm: {doNorm}, doNormForPush: {doNormForPush}")    
    
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
        # we actually used CPU option, GPU one is not well tested

    for deg, degString in degs:

        rootOutPathPrefix = rootOutPathNoMPrefix + "_deg" + degString  # str(deg).replace('.','-')
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

        savePathRoot = os.path.join(rootOutPathPrefix, 'phonetic')

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

                    mapArgs = [(batch[i][:lengths[i]], centers, deg, names[i], outHere, doNorm, doNormForPush, False) for i in range(batch.shape[0])]

                    pool.map(computeAndSaveForLine, mapArgs)

                else:
                    batch = batch.cuda()
                    computeAndSaveForBatch(batch, centers, deg, names, lengths, outHere, doNorm, doNormForPush)


