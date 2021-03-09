from collections import defaultdict
from numba import jit, njit, prange
import numpy as np

def _editdist(s,t,subst_f=1.0):
    dp = [0] * (len(s) + 1)
    d = [0] * (len(s) + 1)
    
    for j in range(1, len(t) + 1):
        d[0] = j
        for i in range(1, len(s) + 1):
            d[i] = min(d[i-1] + 1, dp[i] + 1, dp[i-1] + subst_f * (s[i-1] != t[j-1]))
        dp = list(d)
        d = [0] * (len(s) + 1)

    return min(dp)


def editdist(config):
    swap_cost = config['swap_cost'] if 'swap_cost' in config else 1

    def f(seq, dataset):
        res = defaultdict(int)
        for name, data in dataset:
            res[int(_editdist(data, seq, swap_cost))] += 1
        return str(dict(res)).replace(' ', '')

    return f


def _dtw(s, t, d):
    DTW = [[100000] * (len(t) + 1) for i in range(len(s)+1)]
    
    for i in range(len(s) + 1):
        DTW[i][0] = 0
    
    for i in range(1, len(s)+1):
        for j in range(1, len(t)+1):
            cost = s[i-1] != t[j-1] if d is None else d[s[i-1]][t[j-1]]

            DTW[i][j] = cost + min(DTW[i-1][j], DTW[i][j-1], DTW[i-1][j-1])
    
    return min(DTW[k][len(t)] for k in range(len(s) + 1))


@jit(nopython=True)
def _dtw_numba(s, t, d):
    n = len(t)
    DTW = np.ones((2, n+1)) * 100000
    DTW[0, 0] = 0
    DTW[1, 0] = 0
    q = 1
    best = 100000
    for i in range(len(s)):
        for j in range(n):
            cost = s[i] != t[j] if d is None else d[s[i]][t[j]]
            DTW[q, j+1] = cost + min(DTW[1-q, j+1], DTW[1-q, j], DTW[q, j])
        best = min(best, DTW[q, n])
        q = 1 - q
    return best


@jit(nopython=True)
def _dtw_alt(s, t, d):
    n = len(t)
    DTW = [[100000] * (n + 1) for i in range(2)]
    DTW[0][0] = 0
    DTW[1][0] = 0
    q = 1
    best = 100000
    for i in range(len(s)):
        for j in range(n):
            cost = s[i] != t[j] if d is None else d[s[i]][t[j]]
            DTW[q][j+1] = cost + min(DTW[1-q][j+1], DTW[1-q][j], DTW[q][j])
        best = min(best, DTW[q][n])
        q = 1 - q
    return best


def dtw(config):
    dist = None
    if 'distMatrix' in config:
        dist = []
        for line in open(config['distMatrix'], 'r', encoding='utf8'):
            dist.append(list(map(float, line.strip().split()))) 

    def f(seq, dataset):
        res = defaultdict(int)
        for name, data in dataset:
            res[int(_dtw_numba(data, seq, dist))] += 1
        return str(dict(res)).replace(' ', '')
    return f


def lookup(config):
    mf = config['matchFactors']

    def f(seq, dataset):
        occ = {f:0 for f in mf}
        n = len(seq)
        for _, data_seq in dataset:
            m = len(data_seq)
            for i in range(0, m - n):
                res = sum(data_seq[i+j] == seq[j] for j in range(n))
                for f in mf:
                    if res >= f * n:
                        occ[f] += 1
                    else:
                        break
        return ' '.join(list(str(n) for _, n in sorted(occ.items())))

    return f
