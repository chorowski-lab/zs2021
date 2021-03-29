# %%
import numpy as np
import os
import pathlib
import argparse
import pickle
import yaml
import pandas
import matplotlib.pyplot as plt
from progressbar import ProgressBar, progressbar
from numba import njit, prange
import sys

# %%
parser = argparse.ArgumentParser(description='Compute the pseudo log-proba of a list of sentences')
parser.add_argument('config', type=pathlib.Path, help='Location of the .yaml config file')
parser.add_argument('q', type=int, help='Q')
parser.add_argument('--gold', type=pathlib.Path, default=pathlib.Path('/pio/data/zerospeech2021/dataset/lexical/dev/gold.csv'),
                    help='Location of the gold.csv file')
args = parser.parse_args()


# %%

with open(args.config) as config_file:
    config = yaml.full_load(config_file)


class Dataset:
    def __init__(self, path):
        self.data = []
        self.filenames = []
        self.filename_to_id = dict()
        self.n = 0
        for line in progressbar(open(path, 'r', encoding='utf8')):
            fname, sdesc = line.strip().split()
            self.filenames.append(fname)
            self.filename_to_id[fname] = self.n
            d = list(map(int, sdesc.split(',')))
            self.data.append(np.array(d, dtype='int32'))
            self.n += 1
        self.filenames = np.array(self.filenames)
        self.maxlength = max(len(sample) for sample in self.data)

    def __getitem__(self, i):
        return self.filenames[i], self.data[i]

    def __len__(self):
        return self.n

    def get(self, fname):
        return self.data[self.filename_to_id[fname]]

class Results:
    def __init__(self, path):
        self.filename_to_id = dict()
        self.n = 0
        self.filenames = []
        self.costs = []

        ids = [int(fname.split('-')[1]) for fname in os.listdir(path) if fname.startswith('dev-')]
        n = max(ids)

        if len(set(range(1, n+1)) - set(ids)) > 0:
            raise ValueError(f'some dev-i files are missing')

        for i in range(1, n+1):
            for line in open(path / f'dev-{i}', 'r'):
                fname, costs, Fnames = line.strip().split()
                costs = list(map(float, costs[1:-1].split(',')))
                Fnames = list(map(lambda x: x[1:-1], Fnames[1:-1].split(',')))
                self.costs.append(costs)
                self.filenames.append(Fnames)
                self.filename_to_id[fname] = self.n
                self.n += 1

    def get(self, fname):
        return self.costs[self.filename_to_id[fname]], self.filenames[self.filename_to_id[fname]]

@njit
def dtw_ext(s, t, d):
    n, m = len(t), len(s)
    DTW = np.ones((2, m)) * 100000
    costpath = np.zeros((2, m, n))
    started = np.vstack((np.arange(m), np.zeros(m, dtype='int32')))
    DTW[0, :] = 0
    q = 1
    for i in range(n):
        cost = s[0] != t[i] if d is None else d[s[0], t[i]]
        DTW[q, 0] = DTW[1-q,0] + cost
        if i > 0:
            costpath[q, 0, :i] = costpath[1-q,0,:i]
        costpath[q, 0, i] = cost
        started[1-q,0] = started[q,0]
        for j in range(1, m):
            cost = s[j] != t[i] if d is None else d[s[j], t[i]]
            costpath[q, j, i] = cost
            if DTW[1-q,j-1] <= DTW[1-q, j] and DTW[1-q,j-1] <= DTW[q, j-1]:
                DTW[q,j] = cost + DTW[1-q,j-1]
                costpath[q, j, :i] = costpath[1-q,j-1,:i]
                started[q,j] = started[1-q,j-1]
            elif DTW[1-q,j] <= DTW[1-q, j-1] and DTW[1-q,j] <= DTW[q,j-1]:
                DTW[q,j] = cost + DTW[1-q,j]
                costpath[q, j, :i] = costpath[1-q,j,:i]
                started[q,j] = started[1-q,j]
            else:
                DTW[q,j] = cost + DTW[q,j-1]
                costpath[q, j, :i] = costpath[q,j-1,:i]
                started[q,j] = started[q,j-1]
        q = 1 - q
    bi = np.argmin(DTW[1-q,:])
    return DTW[1-q,bi], costpath[1-q,bi,:], started[1-q,bi], bi

def load_entropy():
    return pickle.load(open('/pio/scratch/2/mstyp/wav2vec/experiments/zerospeech_lm/lstm_3l_qt/entropy/12/entropy', 'rb'))

@njit(parallel=True)
def run(data, lengths, distMatrix, n, maxlength):
    cost_profiles = np.zeros((2, n, maxlength))
    cost_profile_lengths = np.zeros((2, n), dtype='int32')
    ranges = np.zeros((2, n, 2), dtype='int32')
    costs = np.zeros((2, n))
    for i in prange(n):
        top_cost, top_cost_profile, top_a, top_b = dtw_ext(data[i, 1, :lengths[i, 1]], data[i, 0, :lengths[i, 0]], distMatrix)
        mean_cost, mean_cost_profile, mean_a, mean_b = dtw_ext(data[i, 2, :lengths[i, 2]], data[i, 0, :lengths[i, 0]], distMatrix)
        cost_profiles[0, i, :len(top_cost_profile)] = top_cost_profile
        cost_profiles[1, i, :len(mean_cost_profile)] = mean_cost_profile
        cost_profile_lengths[0, i] = len(top_cost_profile)
        cost_profile_lengths[1, i] = len(mean_cost_profile)
        costs[0, i] = top_cost
        costs[1, i] = mean_cost
        ranges[0, i, 0] = top_a
        ranges[0, i, 1] = top_b
        ranges[1, i, 0] = mean_a
        ranges[1, i, 1] = mean_b
    return cost_profiles, cost_profile_lengths, ranges, costs

# %%


trainPath = pathlib.Path(config["trainFile"]) if "trainFile" in config else pathlib.Path(config["trainPath"]) / 'quantized_outputs.txt'
testPath = pathlib.Path(config["testFile"]) if "testFile" in config else pathlib.Path(config["testPath"]) / 'quantized_outputs.txt'
outPath = pathlib.Path(config['outPath']).parents[0]

print('Loading trainset...')
trainset = Dataset(trainPath)
print('Loading testset...')
testset = Dataset(testPath)
print('Loading results...')
results = Results(outPath)

gold = pandas.read_csv(args.gold, header=0).astype({'frequency': pandas.Int64Dtype()})

distMatrix = np.load(config['method']['distMatrix'], allow_pickle=True)

# print('Loading entropy...')

# entropy = load_entropy()

# %%
def load_dtoi():
    if '!info.txt' not in os.listdir(outPath):
        raise ValueError('File !info.txt not found')
    dtoi = dict()
    i = 0
    for line in open(outPath / '!info.txt', 'r'):
        if line.strip().find('-') != -1:
            a, b = map(int, line.strip().split('-'))
            for q in range(a, b+1):
                dtoi[q] = i
                i += 1
        else:
            dtoi[int(line.strip())] = i
            i += 1
    return dtoi

dtoi = load_dtoi()

# %%
print('Preparing data...')

data = np.zeros((len(testset), 3, max(testset.maxlength, trainset.maxlength)), dtype='int32')
lengths = np.zeros((len(testset), 3), dtype='int32')

with ProgressBar(maxval=len(testset)) as bar:
    for i, (fname, testsample) in enumerate(testset):
        bar.update(i)
        costs, Fnames = results.get(fname)
        top_trainsample = trainset.get(Fnames[0])
        mean_trainsample = trainset.get(Fnames[dtoi[args.q]])
        data[i, 0, :len(testsample)] = testsample
        data[i, 1, :len(top_trainsample)] = top_trainsample
        data[i, 2, :len(mean_trainsample)] = mean_trainsample
        lengths[i, 0] = len(testsample)
        lengths[i, 1] = len(top_trainsample)
        lengths[i, 2] = len(mean_trainsample)

# %%

cost_profiles, cost_profile_lengths, ranges, costs = run(data, lengths, distMatrix, len(testset), testset.maxlength)

with open('./results.npy', 'wb') as out:
    np.save(out, cost_profiles)
    np.save(out, cost_profile_lengths)
    np.save(out, ranges)     
    np.save(out, costs)
    
# %%
import numpy as np
with open('/pio/scratch/1/i290956/zs2021/results.npy', 'rb') as f:
    cost_profiles = np.load(f)
    cost_profile_lengths = np.load(f)
    ranges = np.load(f)
    costs = np.load(f)
