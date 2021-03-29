# %%
import numpy as np
import os
import pathlib
import argparse
import pickle
import yaml
import pandas
import matplotlib.pyplot as plt

# %%
parser = argparse.ArgumentParser(description='Compute the pseudo log-proba of a list of sentences')
parser.add_argument('config', type=str, help='Location of the .yaml config file')
parser.add_argument('--gold', type=pathlib.Path, default=pathlib.Path('/pio/data/zerospeech2021/dataset/lexical/dev/gold.csv'),
                    help='Location of the gold.csv file')
args = parser.parse_args()

# %%
class Args:
    config = pathlib.Path('/pio/scratch/1/i290956/zs2021/lexical/configurations/train-960/train-960-dtw-dm-ext.yaml')
    gold = pathlib.Path('/pio/data/zerospeech2021/dataset/lexical/dev/gold.csv')
    q = 500
args = Args()

# %%

with open(args.config) as config_file:
    config = yaml.full_load(config_file)


class Dataset:
    def __init__(self, path):
        self.data = []
        self.filenames = []
        self.filename_to_id = dict()
        self.n = 0
        for line in open(path, 'r', encoding='utf8'):
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

# %%

trainPath = pathlib.Path(config["trainFile"]) if "trainFile" in config else pathlib.Path(config["trainPath"]) / 'quantized_outputs.txt'
testPath = pathlib.Path(config["testFile"]) if "testFile" in config else pathlib.Path(config["testPath"]) / 'quantized_outputs.txt'
outPath = pathlib.Path(config['outPath']).parents[0]

trainset = Dataset(trainPath)
testset = Dataset(testPath)
results = Results(outPath)

gold = pandas.read_csv(args.gold, header=0).astype({'frequency': pandas.Int64Dtype()})

distMatrix = np.load(config['method']['distMatrix'], allow_pickle=True)

entropy = load_entropy()

# %%

distMatrix = np.load('/pio/scratch/1/i290956/zs2021/lexical/dm/distMatrix1.npy')

def gen_profile(fname, offset):
    testsample = testset.get(fname)

    costs, Fnames = results.get(fname)

    trainsample = trainset.get(Fnames[0])

    c, p, a, b = dtw_ext(trainsample, testsample, distMatrix)

    ent = entropy[Fnames[0]]

    # return ent[a-offset:b+1+offset], offset - max(offset-a, 0), b+1-a
    return p

w_colors = ['#00600f', '#6abf69', '#005b9f', '#5eb8ff']
nw_colors = ['#9a0007', '#ff6659', '#bb4d00', '#ffad42']

def generate_plots(id):
    samples = gold[gold['id'] == id]
    words = samples[samples['correct'] == 1]['filename'].to_numpy()
    nonwords = samples[samples['correct'] == 0]['filename'].to_numpy()

    offset = 50
    words_profiles = [gen_profile(fname, offset) for fname in words]
    nonwords_profiles = [gen_profile(fname, offset) for fname in nonwords]

    plt.figure(figsize=(16,9))
    plt.plot(np.arange(len(words_profiles[0])), words_profiles[0], color=w_colors[0])
    plt.plot(np.arange(len(nonwords_profiles[0])), nonwords_profiles[0], color=nw_colors[0])

    # plt.axvspan(off, off+lgh, facecolor='0.2', alpha=0.3)
    plt.xlabel('Time')
    plt.ylabel('Cost')
    plt.plot()

    # plt.close()


generate_plots(5)

# %%
