import yaml
import argparse
import sys
from progressbar import ProgressBar
from collections import defaultdict
from more_itertools import pairwise, grouper
from numba import njit, prange
import numpy as np
import os
import pathlib

def parseArgs():
    parser = argparse.ArgumentParser(description='Compute the pseudo log-proba of a list of sentences')

    parser.add_argument('config', type=str,
                        help='Location of the .yaml config file')

    parser.add_argument('--overwrite', action='store_true', help='Overwrite files if already created')

    parser.add_argument('--debug', type=int,
                        help='run in debug mode, listening on specified port')

    return parser.parse_args()


def parseConfig(args):
    with open(args.config) as config_file:
        return yaml.full_load(config_file)


def squashTransform(data):
    newdata = []
    for x1, x2 in pairwise(data):
        if x1 != x2:
            newdata.append(x1)
    newdata.append(data[-1])
    return newdata


def cleanupTransform(window_size, factor=None, minimum_equal=None):
    def transform(data):
        for i in range(len(data) - window_size + 1):
            if data[i] == data[i+window_size-1]:
                x = data[i]
                cnt = sum(data[j] == x for j in range(i, i+window_size))
                
                if factor is not None and cnt >= window_size * factor:
                    data[i:i+window_size] = [x]*window_size
                elif minimum_equal is not None and cnt >= minimum_equal:
                    data[i:i+window_size] = [x]*window_size
        return data
    return transform


class Dataset:
    def __init__(self, path, transform=None):
        self.data = []
        self.filenames = []
        self.n = 0
        self.transform = transform
        for line in open(path, 'r', encoding='utf8'):
            fname, sdesc = line.strip().split()
            self.filenames.append(fname)
            d = list(map(int, sdesc.split(',')))
            if transform is not None:
                d = transform(d)
            self.data.append(np.array(d, dtype='int32'))
            self.n += 1

    def __getitem__(self, i):
        return self.filenames[i], self.data[i]

    def __len__(self):
        return self.n


class AlignedDataset:
    def __init__(self, path, transform=None):
        _data = []
        self.n = 0
        self.lengths = []
        self.filenames = []
        m = 0
        for line in open(path, 'r', encoding='utf8'):
            fname, sdesc = line.strip().split()
            self.filenames.append(fname)
            d = list(map(int, sdesc.split(',')))
            if transform is not None:
                d = transform(d)
            self.lengths.append(len(d))
            _data.append(np.array(d, dtype='int32'))
            m = max(m, len(d))
            self.n += 1
        self.data = -np.ones((self.n, m), dtype='int32')
        for i in range(self.n):
            self.data[i, :self.lengths[i]] = _data[i]
        self.lengths = np.array(self.lengths, dtype='int32')
        self.filenames = np.array(self.filenames)

    def __getitem__(self, i):
        return self.data[i, :self.lengths[i]]

    def __len__(self):
        return self.n


@njit
def _dtw_numba(s, t, d):
    n = len(t)
    DTW = np.ones((2, n+1)) * 100000
    DTW[0, 0] = 0
    DTW[1, 0] = 0
    q = 1
    best = 100000
    for i in range(len(s)):
        for j in range(n):
            cost = s[i] != t[j] if d is None else d[s[i], t[j]]
            DTW[q, j+1] = cost + min(DTW[1-q, j+1], DTW[1-q, j], DTW[q, j])
        best = min(best, DTW[q, n])
        q = 1 - q
    return best


def dtw(config):
    dist = None
    if 'distMatrix' in config:
        if config['distMatrix'].strip().endswith('.npy'):
            dist = np.load(config['distMatrix'].strip())
        else:
            dist = []
            for line in open(config['distMatrix'], 'r', encoding='utf8'):
                dist.append(list(map(float, line.strip().split()))) 
            dist = np.array(dist)

    @njit(parallel=True)
    def f(seq, dataset, lengths, n):
        res = np.zeros(n)
        for i in prange(n):
            data = dataset[i, :lengths[i]]
            q = _dtw_numba(data, seq, dist)
            res[i] = q
        return res

    return f


def list_to_str(l):
    d = defaultdict(int)
    for i in l:
        d[int(i)] += 1
    return str(dict(d)).replace(' ', '')
    

def run(args, config, method, testset, trainset):
    bar = ProgressBar(maxval=len(testset))
    bar.start()
    i = 0
    for si, subset in enumerate(grouper(testset, config['saveEvery'])):
        path = config["outPath"]+ '-' + str(si+1)
        if os.path.exists(path) and not args.overwrite:
            print(f'File {path} already exists, skipping')
        else:
            with open(path, 'w') as out:
                for index, (fname, seq) in enumerate(subset):
                    bar.update(si*config['saveEvery']+index)
                    res = method(seq, trainset.data, trainset.lengths, trainset.n)
                    out.write(f'{fname} {list_to_str(res)}\n')
    bar.finish()


def extended_run(args, config, ext_depth, method, testset, trainset):
    bar = ProgressBar(maxval=len(testset))
    bar.start()
    i = 0
    indices = slice(ext_depth) if isinstance(ext_depth, int) else ext_depth
    for si, subset in enumerate(grouper(testset, config['saveEvery'])):
        path = config["outPath"]+ '-' + str(si+1)
        if os.path.exists(path) and not args.overwrite:
            print(f'File {path} already exists, skipping')
        else:
            with open(path, 'w') as out:
                for index, (fname, seq) in enumerate(subset):
                    bar.update(si*config['saveEvery']+index)
                    res = method(seq, trainset.data, trainset.lengths, trainset.n)
                    I = np.argsort(res)
                    rs = list(res[I[indices]]).__str__().replace(' ', '')
                    fs = list(trainset.filenames[I[indices]]).__str__().replace(' ', '')
                    out.write(f'{fname} {rs} {fs}\n')
    bar.finish()



def main(args, config):
    outPath = pathlib.Path(config['outPath']) / '..'
    if not os.path.exists(outPath):
        print(f'Output path {config["outPath"]} does not exists, created')
        os.makedirs(outPath)

    transform = None
    if 'transform' in config and config['transform']['name'] == 'squash':
        transform = squashTransform
    
    if 'transform' in config and config['transform']['name'] == 'cleanup':
        transform = cleanupTransform(config['transform']['window_size'], config['transform'].get('factor'), config['transform'].get('minimum_equal'))

    trainPath = config["trainFile"] if "trainFile" in config else f'{config["trainPath"]}/quantized_outputs.txt'
    testPath = config["testFile"] if "testFile" in config else f'{config["testPath"]}/quantized_outputs.txt'

    train = AlignedDataset(trainPath, transform)
    test = Dataset(testPath, transform)

    method = dtw(config['method'])

    if 'extended' in config['method']:
        with open(outPath / '!info.txt', 'w') as out:
            for i in config['method']['extended']:
                out.write(str(i)+'\n')
        extended_run(args, config, config['method']['extended'], method, test, train)
    else:
        run(args, config, method, test, train)


if __name__ == "__main__":
    args = parseArgs()
    if args.debug:
        import ptvsd
        ptvsd.enable_attach(('0.0.0.0', args.debug))
        print("Attach debugger now")
        ptvsd.wait_for_attach()
    config = parseConfig(args)
    main(args, config)
