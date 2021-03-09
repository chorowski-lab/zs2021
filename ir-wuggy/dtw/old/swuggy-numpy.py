# %%
import yaml
import argparse
import sys
import numpy as np
from progressbar import ProgressBar
from dataset import Dataset, AlignedDataset, AlignableTrainset, AlignableTestset
from itertools import tee, zip_longest
from editdist import editdist

def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def parseArgs():
    parser = argparse.ArgumentParser(description='Compute the pseudo log-proba of a list of sentences')

    parser.add_argument('config', type=str,
                        help='Location of the .yaml config file')

    parser.add_argument('--debug', action='store_true',
                        help='run in debug mode')
    return parser.parse_args()

def parseConfig(args):
    with open(args.config) as config_file:
        return yaml.full_load(config_file)

def squashTransform(data: np.array):
    newdata = []
    for x1, x2 in pairwise(data):
        if x1 != x2:
            newdata.append(x1)
    newdata.append(data[-1])
    return np.array(newdata, dtype=data.dtype)


def cleanupTransform(window_size: int, factor: float):
    def transform(data: np.array):
        for i in range(data.shape[0] - window_size + 1):
            if data[i] == data[i+window_size-1]:
                x = data[i]
                cnt = np.count_nonzero(data[i:i+window_size] == x)
                if cnt > window_size * factor:
                    data[i:i+window_size] = i
        return data
    return transform

def computeOccurences(config, seq, dataset):
    mf = config['matchFactors']
    occ = {f:0 for f in mf}
    n = len(seq)
    m = dataset.data.shape[1]
    for i in range(0, m - n):
        res = np.sum(dataset.data[:, i:i+n] == seq, axis=1)
        for f in mf:
            occ[f] += np.sum(res >= f * n)
    return occ


def computePseudoLogProb(config, testset, trainset):
    bar = ProgressBar(maxval=len(testset))
    bar.start()
    for si, subset in enumerate(grouper(testset, config['saveEvery'])):
        with open(f'{config["outPath"]}-{si+1}', 'w') as out:
            for index, (fname, seq) in enumerate(subset):
                bar.update(si*config["saveEvery"]+index)
                occ = computeOccurences(config, seq, trainset)
                occ = list(str(n) for _, n in sorted(occ.items()))
                out.write(f'{fname} {" ".join(occ)}\n')
    bar.finish()

def main(args, config):
    if 'squash' in config:
        transform = squashTransform

    if 'cleanup' in config:
        transform = cleanupTransform(config['cleanup']['window_size'], config['cleanup']['factor'])

    train = Dataset(config['trainPath'], transform)
    test = Dataset(config['testPath'], transform)

    computePseudoLogProb(config, test, train)


if __name__ == "__main__":
    args = parseArgs()
    if args.debug:
        import ptvsd
        ptvsd.enable_attach(('0.0.0.0', 7325))
        print("Attach debugger now")
        ptvsd.wait_for_attach()
    config = parseConfig(args)
    main(args, config)