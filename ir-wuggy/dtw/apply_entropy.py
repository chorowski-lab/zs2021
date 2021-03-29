import yaml
import argparse
import sys
from progressbar import ProgressBar, progressbar
from collections import defaultdict
from more_itertools import pairwise, grouper
from numba import njit, prange
import numpy as np
import os
import pathlib
import pickle


def parseArgs():
    parser = argparse.ArgumentParser(description='Compute the pseudo log-proba of a list of sentences')

    parser.add_argument('config', type=str, help='Location of the .yaml config file')
    parser.add_argument('--dev', action="store_true", help='Use the lexical/dev set')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite files if already created')
    parser.add_argument('--debug', type=int, help='run in debug mode, listening on specified port')
    return parser.parse_args()


def parseConfig(args):
    with open(args.config) as config_file:
        return yaml.full_load(config_file)


def load_entropy():
    return pickle.load(open('/pio/scratch/2/mstyp/wav2vec/experiments/zerospeech_lm/lstm_3l_qt/entropy/12/entropy', 'rb'))


class Dataset:
    def __init__(self, path):
        _data = []
        self.n = 0
        self.lengths = []
        self.filenames = []
        self.filename_to_id = dict()
        m = 0
        for line in progressbar(open(path, 'r', encoding='utf8')):
            fname, sdesc = line.strip().split()
            self.filenames.append(fname)
            self.filename_to_id[fname] = self.n
            d = list(map(int, sdesc.split(',')))
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


    def get_by_filename(self, fname):
        return self[self.filename_to_id[fname]]


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


@njit
def _dtw_ext_numba(s, t, d):
    n, m = len(t), len(s)
    DTW = np.ones((2, m)) * 100000
    started = np.vstack((np.arange(m), np.zeros(m, dtype='int32')))
    DTW[0, :] = 0
    q = 1
    for i in range(n):
        cost = s[0] != t[i] if d is None else d[s[0], t[i]]
        DTW[q, 0] = DTW[1-q,0] + cost
        started[1-q,0] = started[q,0]
        for j in range(1, m):
            cost = s[j] != t[i] if d is None else d[s[j], t[i]]
            if DTW[1-q,j-1] <= DTW[1-q, j] and DTW[1-q,j-1] <= DTW[q, j-1]:
                DTW[q,j] = cost + DTW[1-q,j-1]
                started[q,j] = started[1-q,j-1]
            elif DTW[1-q,j] <= DTW[1-q, j-1] and DTW[1-q,j] <= DTW[q,j-1]:
                DTW[q,j] = cost + DTW[1-q,j]
                started[q,j] = started[1-q,j]
            else:
                DTW[q,j] = cost + DTW[q,j-1]
                started[q,j] = started[q,j-1]
        q = 1 - q
    bi = np.argmin(DTW[1-q,:])
    return DTW[1-q,bi], started[1-q,bi], bi



def main(args, config):
    outPath = (pathlib.Path(config['outPath']) / '..').resolve()
    outPath1 = config["outPath"][:-4] + '-entropy1'
    outPath2 = config["outPath"][:-4] + '-entropy2'
    for path in [outPath1, outPath2]:
        if not os.path.exists(path):
            print(f'Output path {path} does not exists, created')
            os.makedirs(path)

    trainPath = pathlib.Path(config["trainFile"] if "trainFile" in config else f'{config["trainPath"]}/quantized_outputs.txt')
    testPath = pathlib.Path(config["testFile"] if "testFile" in config else f'{config["testPath"]}/quantized_outputs.txt')

    trainset = Dataset(trainPath)
    testset = Dataset(testPath)

    test_to_train_ids = np.zeros((len(testset)), dtype='int32')

    entropy = load_entropy()

    ids = [int(fname.split('-')[1]) for fname in os.listdir(outPath) if fname.startswith('dev-')]
    n = max(ids)

    if len(set(range(1, n+1)) - set(ids)) > 0:
        raise ValueError(f'some dev-i files are missing')

    print('Aligning train set to test set')
    bar = ProgressBar(maxval=n)
    bar.start()
    for i in range(1, n+1):
        bar.update(i-1)
        for line in open(outPath / f'dev-{i}', 'r'):
            fname, costs, Fnames = line.strip().split()
            costs = list(map(float, costs[1:-1].split(',')))
            Fnames = list(map(lambda x: x[1:-1], Fnames[1:-1].split(',')))
            test_to_train_ids[testset.filename_to_id[fname]] = trainset.filename_to_id[Fnames[0]]
    bar.finish()
    
    dist = None
    if 'distMatrix' in config:
        if config['distMatrix'].strip().endswith('.npy'):
            dist = np.load(config['distMatrix'].strip())
        else:
            dist = []
            for line in open(config['distMatrix'], 'r', encoding='utf8'):
                dist.append(list(map(float, line.strip().split()))) 
            dist = np.array(dist)

    bar = ProgressBar(maxval=len(testset))
    bar.start()
    i = 0
    for si, subset in enumerate(grouper(testset, config['saveEvery'])):
        path = outPath1 + '/dev-' + str(si+1)
        if os.path.exists(path) and not args.overwrite:
            print(f'File {path} already exists, skipping')
        else:
            with open(path, 'w') as out:
                for index, seq in enumerate(subset):
                    j = si*config['saveEvery']+index
                    fname = testset.filenames[j]
                    bar.update(j)
                    score, a, b = _dtw_ext_numba(seq, trainset[test_to_train_ids[j]], dist)
                    Fname = trainset.filenames[test_to_train_ids[j]]
                    for i in range(a, b):
                        score += ((i - a) / (b - a)) ** 2 * entropy[Fname][i]
                    out.write(f'{fname} [{score}] [\'STUB\']\n')
    bar.finish()

    bar = ProgressBar(maxval=len(testset))
    bar.start()
    i = 0
    for si, subset in enumerate(grouper(testset, config['saveEvery'])):
        path = outPath2 + '/dev-' + str(si+1)
        if os.path.exists(path) and not args.overwrite:
            print(f'File {path} already exists, skipping')
        else:
            with open(path, 'w') as out:
                for index, seq in enumerate(subset):
                    j = si*config['saveEvery']+index
                    fname = testset.filenames[j]
                    bar.update(j)
                    score, a, b = _dtw_ext_numba(seq, trainset[test_to_train_ids[j]], dist)
                    Fname = trainset.filenames[test_to_train_ids[j]]
                    for i in range(a, b):
                        score -= ((i - a) / (b - a)) ** 2 * entropy[Fname][i]
                    out.write(f'{fname} [{score}] [\'STUB\']\n')
    bar.finish()

if __name__ == "__main__":
    args = parseArgs()
    if args.debug:
        import ptvsd
        ptvsd.enable_attach(('0.0.0.0', args.debug))
        print("Attach debugger now")
        ptvsd.wait_for_attach()
    config = parseConfig(args)
    main(args, config)