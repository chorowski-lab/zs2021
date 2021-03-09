from collections import defaultdict
from progressbar import ProgressBar
import numpy as np

trainPath = '/pio/data/zerospeech2021/quantized/LibriSpeech/dev-clean'
testPath = '/pio/data/zerospeech2021/quantized/lexical/dev-big-no-oov'


lls = defaultdict(int)
min_l = 1000
max_l = 0
average_l = 0


bar = ProgressBar(max_value=40000)
for line in open(f'{testPath}/quantized_outputs.txt', 'r', encoding='utf8'):
    fname, sdesc = line.strip().split()
    l = len(sdesc.split(','))
    min_l = min(min_l, l)
    max_l = max(max_l, l)
    lls[l] += 1
    average_l += l

average_l /= 40000
print(average_l)
# print(min_l, max_l)
# print(lls)


min_l = 1000
max_l = 0
lines = 0

subseqs = 0
subseqs_l = 0

tr_lls = defaultdict(int)
for line in open(f'{trainPath}/quantized_outputs.txt', 'r', encoding='utf8'):
    fname, sdesc = line.strip().split()
    l = len(sdesc.split(','))
    # tr_lls[l] += 1
    subseqs += (l + 1) * l / 2
    subseqs_l += l / 2

print(subseqs)
print(subseqs_l / 2703)


def compute_v_size(l):
    sz = 0
    for trl in tr_lls:
        if trl >= l:
           sz += tr_lls[trl] * (trl - l + 1)
    return sz


bar = ProgressBar(max_value=2703*len(lls))
bi = 0

# for l in lls:
#     sz = compute_v_size(l)
#     data = np.zeros((sz, l))
#     k = 0
#     for line in open(f'{trainPath}/quantized_outputs.txt', 'r', encoding='utf8'):
#         bar.update(bi)
#         bi += 1
#         fname, sdesc = line.strip().split()
#         d = np.array(list(int(x) for x in sdesc.split(',')))
#         for i in range(len(d) - l + 1):
#             data[k, :] = d[i:i+l]
#             k += 1
# bar.finish()