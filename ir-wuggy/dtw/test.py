import numpy as np

d = np.array([list(map(float, line.strip().split())) for line in open('/pio/scratch/1/i290956/zs2021/quantized/ppusz/distMatrix.txt', 'r')])
d[d < 0] = 0

m = np.zeros_like(d)
for i in range(50):
    for j in range(50):
        m[i,j] = (d[1,j] ** 2 + d[i,1] ** 2 - d[i,j] ** 2) / 2

print(m)
