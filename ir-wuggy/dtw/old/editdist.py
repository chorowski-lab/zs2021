import numpy as np

def editdist(s, t, subst_f=1.0):

    d = np.zeros((len(s) + 1, len(t) + 1))
    for j in range(len(t) + 1):
        d[0,j] = j
    
    for j in range(1, len(t) + 1):
        for i in range(1, len(s) + 1):

            d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + subst_f * (s[i-1] != t[j-1]))
    
    return d[:, len(t)].min()
