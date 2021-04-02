import numpy as np
import pickle




dm2 = np.zeros((50, 50))
dm1 = np.zeros((50, 50))

for i, line in enumerate(open('/pio/scratch/1/i283340/MGR/zs/quantization/nullspCosineClustersCosSqDM/distMatrix.txt', 'r')):
    for j, x in enumerate(map(float, line.strip().split())):
        dm2[i, j] = max(x, 0)

for i, line in enumerate(open('/pio/scratch/1/i283340/MGR/zs/quantization/nullspCosineClustersCosDM/distMatrix.txt', 'r')):
    for j, x in enumerate(map(float, line.strip().split())):
        dm1[i, j] = max(x, 0)

dm1 = np.load('/pio/scratch/1/i290956/zs2021/lexical/dm/distMatrix1.npy')

print(dm1)

for i in range(9):
    np.save(f'/pio/scratch/1/i290956/zs2021/lexical/dm/distMatrix1_{i+1}', np.power(dm1, 1 + (i+1)/10))


# np.save('/pio/scratch/1/i290956/zs2021/lexical/dm/distMatrixCos1', dm1)
# np.save('/pio/scratch/1/i290956/zs2021/lexical/dm/distMatrixCos1_6', np.power(dm1, 1.6))
# np.save('/pio/scratch/1/i290956/zs2021/lexical/dm/distMatrixCos1_7', np.power(dm1, 1.7))
# np.save('/pio/scratch/1/i290956/zs2021/lexical/dm/distMatrixCos2', dm2)
# np.save('/pio/scratch/1/i290956/zs2021/lexical/dm/distMatrixCos2_5', dm2_5)
# np.save('/pio/scratch/1/i290956/zs2021/lexical/dm/distMatrixCos3', dm3)
# np.save('/pio/scratch/1/i290956/zs2021/lexical/dm/distMatrixCos4', dm4)
# np.save('/pio/scratch/1/i290956/zs2021/lexical/dm/distMatrixCos5', dm5)
# np.save('/pio/scratch/1/i290956/zs2021/lexical/dm/distMatrixCos6', dm6)
# np.save('/pio/scratch/1/i290956/zs2021/lexical/dm/distMatrixCos7', dm7)
# np.save('/pio/scratch/1/i290956/zs2021/lexical/dm/distMatrixCos8', dm8)
