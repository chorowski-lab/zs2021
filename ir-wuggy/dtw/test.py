import numpy as np
from time import sleep

def t1():
    d = np.array([list(map(float, line.strip().split())) for line in open('/pio/scratch/1/i290956/zs2021/quantized/ppusz/distMatrix.txt', 'r')])
    d[d < 0] = 0

    m = np.zeros_like(d)
    for i in range(50):
        for j in range(50):
            m[i,j] = (d[1,j] ** 2 + d[i,1] ** 2 - d[i,j] ** 2) / 2

    print(m)

def t2():
    n = 0 
    for line in open('/pio/data/zerospeech2021/quantized/LibriSpeech/train-full-960/quantized_outputs.txt', 'r'):
        n += 1
    print(n)



def t3():

    prefix = '/pio/scratch/1/i290956/zs2021/output/lexical/train-full-960'
    with open(f'{prefix}/dtw-dm-ext/dev-1', 'w') as out:
        for i in range(1, 41):
            for l1, l2 in zip(open(f'{prefix}/dtw-dm-xx/dev-{i}'), open(f'{prefix}/dtw-dm-xxx/dev-{i}')):
                fname, d1, f1 = l1.split()
                fname, d2, f2 = l2.split()
                out.write(f'{fname} {d1[:-1]},{d2[1:]} {f1[:-1]},{f2[1:]}\n')

def t4():
    prefix = '/pio/scratch/1/i290956/zs2021/output/lexical/train-full-960'
    for line in open(f'{prefix}/dtw-dm-ext/dev-1', 'r'):
        fname, desc, fnames = line.split()
        d = list(map(float, desc[1:-1].split(',')))
        f = fnames[1:-1].split(',')
        print(d)
        print(len(d))
        print(len(f))
        break

def t5():
    with open('/pio/data/zerospeech2021/quantized/lexical/test-120000/quantized_outputs.txt', 'w') as out:
        for i, line in enumerate(open('/pio/data/zerospeech2021/quantized/lexical/test/quantized_outputs.txt', 'r')):
            if i >= 200000:
                out.write(line.strip() + '\n')


def t6():
    with open('/pio/scratch/1/i290956/zs2021/test', 'w') as out:
        while True:
            sleep(10)
            out.write('test\n')
            print('test')

def t7():
    for i in range(1, 41):
        with open(f'/pio/scratch/1/i290956/zs2021/output/lexical/train-full-960/dtw-dm-ext/dev-{i}', 'w') as out:
            for line_1, line_2 in zip(open(f'/pio/scratch/1/i290956/zs2021/output/lexical/train-full-960/dtw-dm-xx/dev-{i}', 'r'), open(f'/pio/scratch/1/i290956/zs2021/output/lexical/train-full-960/dtw-dm-xxx/dev-{i}', 'r')):
                fname_1, desc_1, fs_1 = line_1.split()
                fname_2, desc_2, fs_2 = line_2.split()
                desc = desc_1[:-1]+','+desc_2[1:]
                fs = fs_1[:-1]+','+fs_2[1:]
                out.write(f'{fname_1} {desc} {fs}\n')
                

with open('./test', 'w') as out:
    for i in range(1, 1001):
        out.write(f'{i}\n')
        