import sys
import os
import numpy as np
import re
import random

class DSgetter:

    def __init__(self, path, getFromFileMethod, nameMustMatch=None, nameMustNotMatch=None):
        self.root = path
        self.getFromFileMethod = getFromFileMethod
        self.nameMustMatch = nameMustMatch
        self.nameMustNotMatch = nameMustNotMatch
        self.reprSize = None
        self.files = []
        #self.sizes = []
        self.where = {}
        #self.size = 0
        pattern = re.compile(nameMustMatch) if nameMustMatch is not None else None
        badPattern = re.compile(nameMustNotMatch) if nameMustNotMatch is not None else None
        random.seed(12345)
        walk = list([(a,b,d) for a,b,c in os.walk(path) for d in c])  #list(os.walk(path))
        random.shuffle(walk)  # same random shuffling
        #for p,sd,f in walk: #sorted(os.walk(path)): #scandir(sys.argv[1]):
        #    for name in sorted(f):
        for p, sd, name in walk:
            if (nameMustMatch is None or pattern.match(name)) \
            and (nameMustNotMatch is None or not badPattern.match(name)): 
                fn = os.path.join(p, name)
                #arr = self.getFromFileMethod(fn)  #np.genfromtxt(fn)
                self.files.append((p, name)) #, arr.shape[0]))
                #self.sizes.append(arr.shape[0])
                assert name not in self.where
                self.where[name] = len(self.files) - 1  # assumes names are directory-disjoint
                #self.size += arr.shape[0]
                #assert self.reprSize is None or self.reprSize == arr.shape[1]
                #self.reprSize = arr.shape[1]
        print("Files in DS:", len(self.files))
        #assert self.reprSize is not None

    def getFromFile(self, fname):
        if fname not in self.where:
            return None
        p,n = self.files[self.where[fname]]
        return self.getFromFileMethod(os.path.join(p, n))

    def batchGenGen(self, filesInBatch=4):
        return lambda: self.batchGenerator(filesInBatch=filesInBatch)

    def batchGenerator(self, filesInBatch=4):
        for begin in range(0, len(self.files), filesInBatch):
            batch = []
            names = []
            lengths = []
            size = 0
            for i in range(begin, min(begin + filesInBatch, len(self.files))):
                p,n = self.files[i]
                arr = self.getFromFileMethod(os.path.join(p, n))
                if self.reprSize is None:
                    self.reprSize = arr.shape[1]
                assert self.reprSize == arr.shape[1]
                batch.append(arr)
                size += arr.shape[0]  #self.sizes[i]
                names.append(n)
                lengths.append(arr.shape[0])  #self.sizes[i])
            batchnp = np.zeros((len(batch), max([l.shape[0] for l in batch]), self.reprSize))
            padMask = np.full((len(batch), max([l.shape[0] for l in batch])), False, dtype=bool)
            sumlen = 0
            for i, l in enumerate(batch):
                batchnp[i][:l.shape[0]] = l
                padMask[i][l.shape[0]:] = True
            yield (batchnp, padMask, size, lengths, names)

    def randomChoice(self, n):
        choice = sorted(random.sample(range(0, self.size), n))
        arr = []
        i = 0
        cursize = 0
        p,n = self.files[i]
        arr1 = self.getFromFileMethod(os.path.join(p, n))
        for num in choice:
            while cursize + arr1.shape[0] <= num:
                cursize += arr1.shape[0]
                i += 1
                p,n = self.files[i]
                arr1 = self.getFromFileMethod(os.path.join(p, n))
            arr.append(arr1[num - cursize])
        return np.array(arr)


