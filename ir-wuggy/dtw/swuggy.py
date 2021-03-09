import yaml
import argparse
import sys
from progressbar import ProgressBar
from src.dataset import Dataset, AlignedDataset
from src.methods import editdist, lookup, dtw
from collections import defaultdict
from more_itertools import pairwise, grouper
from numba import njit, prange

def parseArgs():
    parser = argparse.ArgumentParser(description='Compute the pseudo log-proba of a list of sentences')

    parser.add_argument('config', type=str,
                        help='Location of the .yaml config file')
    parser.add_argument('--debug', type=int,
                        help='run in debug mode, listening on specified port')

    parser.add_argument('--extended', action='store_true',
                        help='Run in the extended mode - list all distances (no rounding/gathering to dict), remember also 100 best matches')
                        
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


def cleanupTransform(window_size, factor = 0.5):
    def transform(data):
        for i in range(len(data) - window_size + 1):
            if data[i] == data[i+window_size-1]:
                x = data[i]
                cnt = sum(data[j] == x for j in range(i, i+window_size))
                if cnt >= window_size * factor:
                    data[i:i+window_size] = [x]*window_size
        return data
    return transform


@njit(parallel=True)
def run_subset(subset, bar, i, method, trainset):
    results = ''
    for k in prange(len(subset)):
        fname, seq = subset[k]
        res = method(seq, trainset)
        results += fname + ' ' + res + '\n'
        bar(i)
        i += 1
    return results


def run(config, method, testset, trainset):
    bar = ProgressBar(maxval=len(testset))
    bar.start()
    i = 0
    for si, subset in enumerate(grouper(testset, config['saveEvery'])):
            # for index, (fname, seq) in enumerate(subset):
            #     bar.update(si*config['saveEvery']+index)
            #     res = method(seq, trainset)
            #     out.write(f'{fname} {res}\n')
        results = run_subset(subset, lambda i: bar.update(i), si*config['saveEvery'], method, trainset)

        with open(config["outPath"]+ '-' + str(si+1), 'w') as out:
            out.write(results)


    bar.finish()


def main(args, config):

    transform = None
    if 'transform' in config and config['transform']['name'] == 'squash':
        transform = squashTransform
    
    if 'transform' in config and config['transform']['name'] == 'cleanup':
        transform = cleanupTransform(config['transform']['window_size'], config['transform']['factor'])

    trainPath = config["trainFile"] if "trainFile" in config else f'{config["trainPath"]}/quantized_outputs.txt'
    testPath = config["testFile"] if "testFile" in config else f'{config["testPath"]}/quantized_outputs.txt'

    train = AlignedDataset(trainPath, transform)
    test = Dataset(testPath, transform)

    method = None
    if config['method']['name'] == 'editdist':
        method = editdist(config['method'])

    if config['method']['name'] == 'lookup':
        method = lookup(config['method'])
    
    if config['method']['name'] == 'dtw':
        method = dtw(config['method'])

    run(config, method, test, train)


if __name__ == "__main__":
    args = parseArgs()
    if args.debug:
        import ptvsd
        ptvsd.enable_attach(('0.0.0.0', args.debug))
        print("Attach debugger now")
        ptvsd.wait_for_attach()
    config = parseConfig(args)
    main(args, config)