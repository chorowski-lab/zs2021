


import sys
import os
from shutil import copyfile

# example:  python clean_phonetic_quantized_to_dir.py /pio/data/zerospeech2021/dataset/phonetic/dev-clean ../quantizedtoomuch ../quantizedcleanout quantized_outputs.txt

dsRoot = sys.argv[1]  # data files need to be flat just under here
quantizedRoot = sys.argv[2]  # quantizedFName needs to be just there
newQuantizedRoot = sys.argv[3]
quantizedFName = sys.argv[4]

DSfileset = set()

DSfiles = [fn.split('.')[0] for fn in os.listdir(dsRoot) if os.path.isfile(os.path.join(dsRoot, fn)) and fn[0].isdigit()]

for fn in DSfiles:
    DSfileset.add(fn)

print(f'files in ds: {len(DSfileset)}')

badFiles = 0
okFiles = 0

os.makedirs(newQuantizedRoot)

open(os.path.join(newQuantizedRoot, quantizedFName), 'w').close()

with open(os.path.join(quantizedRoot, quantizedFName), 'r') as f:
    with open(os.path.join(newQuantizedRoot, quantizedFName), 'a') as qf:
        for line in f:
            data = line.split('\t')
            fname = data[0]
            if fname not in DSfileset:
                badFiles += 1
                continue
            okFiles += 1
            qf.write(line)

print(f'files ok: {okFiles}, bad files: {badFiles}')

