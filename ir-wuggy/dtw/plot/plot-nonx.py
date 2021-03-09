import matplotlib.pyplot as plt
from os import listdir, mkdir
import pandas 
import sys
import numpy as np
import pathlib
import argparse
from progressbar import ProgressBar
from more_itertools import take

np.random.seed(873382376)

# parser = argparse.ArgumentParser()

# parser.add_argument('word_ids', type=str,
#                     help='Id\'s of the words for which the plots should be prepared.')

# args = parser.parse_args()

zs2021 = '/pio/scratch/1/i290956/zs2021'
store = f'{zs2021}/output/lexical/train-full-960/dtw-dm'
gold = pandas.read_csv(f'{zs2021}/dataset/lexical/gold.csv', \
    header=0, index_col='filename').astype({'frequency': pandas.Int64Dtype()})

ids = [int(fname.split('-')[1]) for fname in listdir(store) if fname.startswith('dev-')]

results = pandas.concat(
    pandas.read_csv(f'{store}/dev-{i}', sep=' ', header=None, \
        names=['filename', 'result'], index_col='filename') \
    for i in ids)

data = pandas.concat([gold, results], axis=1)
data.reset_index(inplace=True)
data.drop(columns=['phones', 'filename'], inplace=True)
data = data[data['result'].notna()]


def get_range(samples):
    rmin, rmax = 10000, 0
    for res in samples['result']:
        for k, _ in eval(res).items():
            rmin = min(rmin, k)
            rmax = max(rmax, k)
    return rmin, rmax


def minf(res):
    return - min(k for k in eval(res).keys())


def meanf(res):
    return - sum(k * v for k, v in eval(res).items()) # / sum(v for v in eval(res).values()) - constant


def get_score(samples, func):
    word_samples = samples[samples['correct'] == 1]
    nonword_samples = samples[samples['correct'] == 0]

    samples = pandas.concat([
        samples.loc[samples['correct'] == 1].reset_index().rename(
            lambda x: 'w_' + x, axis=1),
        samples.loc[samples['correct'] == 0].reset_index().rename(
            lambda x: 'nw_' + x, axis=1)], axis=1)
    
    results = samples[['w_result', 'nw_result']].applymap(func).to_numpy().T
    score = np.mean((results[0,:] == results[1,:]) *.5 + (results[0,:] > results[1,:]))
    return score, results


def get_word_nonword(samples):
    word = samples[samples['correct'] == 1]['word'].iloc[0]
    nonword = samples[samples['correct'] == 0]['word'].iloc[0]
    return word, nonword


w_colors = ['#00600f', '#6abf69', '#005b9f', '#5eb8ff']
nw_colors = ['#9a0007', '#ff6659', '#bb4d00', '#ffad42']

def to_ys_array(samples):
    rmin, rmax = get_range(samples)
    xs = np.arange(rmin, rmax + 1)

    word_samples = samples[samples['correct'] == 1]
    nonword_samples = samples[samples['correct'] == 0]

    w_ys = np.zeros((len(word_samples), rmax - rmin + 1), dtype='int32')
    nw_ys = np.zeros((len(nonword_samples), rmax - rmin + 1), dtype='int32')

    for i, w_res in enumerate(word_samples['result']):
        for k, v in eval(w_res).items():
            w_ys[i, k-rmin] = v
    
    for i, nw_res in enumerate(nonword_samples['result']):
        for k, v in eval(nw_res).items():
            nw_ys[i, k-rmin] = v
    
    return xs, w_ys, nw_ys


def to_lists(samples, size):
    ls = [[] for _ in range(len(samples))]
    for i, res in enumerate(samples['result']):
        for k, v in eval(res).items():
            for _ in range(v):
                ls[i].append(int(k))
    return list(list(sorted(l))[:size] for l in ls)


def to_new_ys_array(samples):
    rmin, rmax = 10000, 0
    for res in samples['result']:
        for k, _ in eval(res).items():
            rmin = min(rmin, k)
            rmax = max(rmax, k)
    return rmin, rmax

def plot_average(path, samples, log=False):
    minscore, minresults = get_score(samples, minf)
    meanscore, meanresults = get_score(samples, meanf)

    word, nonword = get_word_nonword(samples)    

    xs, w_ys, nw_ys = to_ys_array(samples)

    plt.figure(figsize=(16,9))
    plt.title(f'word-nonword: {word}-{nonword}; by voice; min score: {minscore}; mean score: {meanscore}')
    
    plt.plot(xs, np.mean(w_ys, axis=0), color='green', label=f'W: {word}')
    plt.plot(xs, np.mean(nw_ys, axis=0), color='red', label=f'NW: {nonword}')
    
    plt.fill_between(xs, np.min(w_ys, axis=0), np.max(w_ys, axis=0), alpha=.1, color='green')
    plt.fill_between(xs, np.min(nw_ys, axis=0), np.max(nw_ys, axis=0), alpha=.1, color='red')

    plt.xlabel('DTW cost')
    plt.ylabel('No. of occurences')
    plt.legend()
    plt.savefig(f'{path}/{word}-{nonword}_average{"_logscale" if log else ""}.svg')
    plt.close()

def plot_by_voice(path, samples, log=False):
    minscore, minresults = get_score(samples, minf)
    meanscore, meanresults = get_score(samples, meanf)

    word, nonword = get_word_nonword(samples)    

    xs, w_ys, nw_ys = to_ys_array(samples)

    plt.figure(figsize=(16,9))
    plt.title(f'word-nonword: {word}-{nonword}; by voice; min score: {minscore}; mean score: {meanscore}')

    for i in range(4):
        plt.plot(xs, w_ys[i,:], color=w_colors[i%4], label=f'{"ACDF"[i]} (min: {-minresults[0,i]})')
         
    for i in range(4):
        plt.plot(xs, nw_ys[i,:], color=nw_colors[i%4], label=f'{"ACDF"[i]} (min: {-minresults[1,i]})')
    
    if log:
        plt.yscale('log')

    plt.xlabel('DTW cost')
    plt.ylabel('No. of occurences')
    plt.legend(ncol=2)
    plt.savefig(f'{path}/{word}-{nonword}_by_voice{"_logscale" if log else ""}.svg')
    plt.close()


def plot_linear(path, samples, size=1000):
    minscore, minresults = get_score(samples, minf)
    meanscore, meanresults = get_score(samples, meanf)

    word, nonword = get_word_nonword(samples)    

    w_ys = to_lists(samples[samples['correct'] == 1], size)
    nw_ys = to_lists(samples[samples['correct'] == 0], size)
    xs = np.arange(size)

    plt.figure(figsize=(16,9))
    plt.title(f'word-nonword: {word}-{nonword}; min score: {minscore}; mean score: {meanscore}')

    for i in range(4):
        plt.plot(xs, w_ys[i], color=w_colors[i%4], label=f'{"ACDF"[i]} (min: {-minresults[0,i]})')
         
    for i in range(4):
        plt.plot(xs, nw_ys[i],  color=nw_colors[i%4], label=f'{"ACDF"[i]} (min: {-minresults[1,i]})')
    
    plt.ylabel('DTW cost')
    plt.legend(ncol=2)
    plt.savefig(f'{path}/{word}-{nonword}_s{size}_c.svg')
    plt.close()



def run_random(n):
    bar = ProgressBar(maxval=n)
    bar.start()
    for i, id in enumerate(np.random.choice(5588, size=n, replace=False)):
        bar.update(i)
        samples = data[data['id'] == id]
        if len(samples) > 0:
            minscore, minresults = get_score(samples, minf)
            meanscore, meanresults = get_score(samples, meanf)
            word, nonword = get_word_nonword(samples)
            path = f'{zs2021}/plots/v1/min{str(minscore)}_mean{str(meanscore)}_{word}-{nonword}'
            # path = f'{zs2021}/plots/tmp'
            try:
                mkdir(path)
            except FileExistsError:
                pass
            # plot_average(path, samples)
            # plot_average(path, samples, True)
            # plot_by_voice(path, samples)
            # plot_by_voice(path, samples, True)
            plot_linear(path, samples, 100)
            plot_linear(path, samples, 250)
            plot_linear(path, samples)
    bar.finish()


run_random(100)

 