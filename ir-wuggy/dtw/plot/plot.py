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
store = f'{zs2021}/output/lexical/train-full-960/dtw-dm-x'
gold = pandas.read_csv(f'{zs2021}/dataset/lexical/gold.csv', \
    header=0, index_col='filename').astype({'frequency': pandas.Int64Dtype()})

ids = [int(fname.split('-')[1]) for fname in listdir(store) if fname.startswith('dev-')]

results = pandas.concat(
    pandas.read_csv(f'{store}/dev-{i}', sep=' ', header=None, \
        names=['filename', 'result', 'filenames'], index_col='filename') \
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
    return - eval(res)[0]


def get_min_score(samples):
    word_samples = samples[samples['correct'] == 1]
    nonword_samples = samples[samples['correct'] == 0]

    samples = pandas.concat([
        samples.loc[samples['correct'] == 1].reset_index().rename(
            lambda x: 'w_' + x, axis=1),
        samples.loc[samples['correct'] == 0].reset_index().rename(
            lambda x: 'nw_' + x, axis=1)], axis=1)
    
    results = samples[['w_result', 'nw_result']].applymap(minf).to_numpy().T
    score = np.mean((results[0,:] == results[1,:]) *.5 + (results[0,:] > results[1,:]))
    return score, results


def get_word_nonword(samples):
    word = samples[samples['correct'] == 1]['word'].iloc[0]
    nonword = samples[samples['correct'] == 0]['word'].iloc[0]
    return word, nonword


w_colors = ['#00600f', '#6abf69', '#005b9f', '#5eb8ff']
nw_colors = ['#9a0007', '#ff6659', '#bb4d00', '#ffad42']



def plot_linear(path, samples, size=1000):
    minscore, minresults = get_min_score(samples)
    word, nonword = get_word_nonword(samples)    

    w_ys = [eval(res)[:size] for res in samples[samples['correct'] == 1]['result']]
    nw_ys = [eval(res)[:size] for res in samples[samples['correct'] == 0]['result']]
    xs = np.arange(size)

    plt.figure(figsize=(16,9))
    plt.title(f'word-nonword: {word}-{nonword}; min score: {minscore}')

    for i in range(4):
        plt.plot(xs, w_ys[i], color=w_colors[i%4], label=f'{"ACDF"[i]} (min: {-minresults[0,i]})')
         
    for i in range(4):
        plt.plot(xs, nw_ys[i],  color=nw_colors[i%4], label=f'{"ACDF"[i]} (min: {-minresults[1,i]})')
    
    plt.ylabel('DTW cost')
    plt.legend(ncol=2)
    plt.savefig(f'{path}/{word}-{nonword}_s{size}_c.svg')
    plt.close()


def run():
    bar = ProgressBar(maxval=250)
    bar.start()
    for i, id in enumerate(set(data['id'].values)):
        bar.update(i)
        samples = data[data['id'] == id]
        word, nonword = get_word_nonword(samples)
        minscore, minresults = get_min_score(samples)

        path = f'{zs2021}/plots/v2/min{str(minscore)}_{word}-{nonword}'
        # path = f'{zs2021}/plots/tmp'
        try:
            mkdir(path)
        except FileExistsError:
            pass
        plot_linear(path, samples, 10)
        plot_linear(path, samples, 50)
        # plot_linear(path, samples, 100)
        # plot_linear(path, samples, 250)
        # plot_linear(path, samples)
    bar.finish()

run()
 