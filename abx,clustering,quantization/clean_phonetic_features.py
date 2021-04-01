


import sys
import os
from shutil import copyfile

# example:  python clean_phonetic_features_to_dir.py /pio/data/zerospeech2021/dataset/phonetic/dev-clean ../features_lvl2_all-ls_devclean/ ../features_lvl2_all-ls_devclean_zscleaned/

dsRoot = sys.argv[1]  # data files need to be flat just under here
featureRoot = sys.argv[2]  # feature files need to be flat just under here
#newFeatureRoot = sys.argv[3]

DSfileset = set()

DSfiles = [fn.split('.')[0] for fn in os.listdir(dsRoot) if os.path.isfile(os.path.join(dsRoot, fn)) and fn[0].isdigit()]

for fn in DSfiles:
    DSfileset.add(fn)

print(f'files in ds: {len(DSfileset)}')

featureFilesOk = [fn for fn in os.listdir(featureRoot) if os.path.isfile(os.path.join(featureRoot, fn)) and fn.split('.')[0] in DSfileset]
featureFilesBad = [fn for fn in os.listdir(featureRoot) if os.path.isfile(os.path.join(featureRoot, fn)) and not fn.split('.')[0] in DSfileset]

print(f'Ok feature files: {len(featureFilesOk)}, extra feature files to be removed: {len(featureFilesBad)}')

#os.makedirs(newFeatureRoot)

i = 0
for fn in featureFilesBad:  #featureFilesOk:
    #copyfile(os.path.join(featureRoot, fn), os.path.join(newFeatureRoot, fn))
    os.remove(os.path.join(featureRoot, fn))
    i += 1
    if i % 100 == 0:
        print(f'{i} feature files copied')

