import argparse
from os import listdir
import sys


parser = argparse.ArgumentParser(description='Compute the pseudo log-proba of a list of sentences')
parser.add_argument('data', type=str, help='Path to the folder containing the "dev-i" files')
parser.add_argument('--test', action="store_true", help='Whether to use test or dev set')
args = parser.parse_args()

ids = [int(fname.split('-')[1]) for fname in listdir(args.data) if fname.startswith('dev-')]
n = max(ids)

if len(set(range(1, n+1)) - set(ids)) > 0:
    raise ValueError(f'some dev-i files are missing')

expected_fnames = set()

for line in open(f'/pio/data/zerospeech2021/quantized/lexical/{"test" if args.test else "dev-big-no-oov"}/quantized_outputs.txt', 'r'):
    fname = line.split()[0]
    expected_fnames.add(fname)

fnames = set()

for i in range(1, n+1):
    for line in open(f'{args.data}/dev-{i}', 'r'):
        fname = line.split()[0]
        fnames.add(fname)

if len(expected_fnames - fnames) > 0:
    print('Some values are missing!')
elif len(fnames - expected_fnames) > 0:
    print('There is too much data!')
else:
    print('OK')
