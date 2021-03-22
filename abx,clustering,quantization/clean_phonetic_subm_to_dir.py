


import sys
import os
from shutil import copyfile

dsRoot = sys.argv[1]
submRoot = sys.argv[2]
newSubmRoot = sys.argv[3]
metadataFileToCpy = sys.argv[4]



sDevClean = set()
sDevOther = set()

dirFilesDevClean = [fn.split('.')[0] for fn in os.listdir(os.path.join(dsRoot, "phonetic/dev-clean")) if os.path.isfile(os.path.join(dsRoot, "phonetic/dev-clean", fn)) and not fn.startswith('dev')]
dirFilesDevOther = [fn.split('.')[0] for fn in os.listdir(os.path.join(dsRoot, "phonetic/dev-other")) if os.path.isfile(os.path.join(dsRoot, "phonetic/dev-other", fn)) and not fn.startswith('dev')]

for fn in dirFilesDevClean:
    sDevClean.add(fn)

for fn in dirFilesDevOther:
    sDevOther.add(fn)

print(dirFilesDevClean[:10])
print(dirFilesDevOther[:10])

# print(len(s))

submFilesDevCleanOk = [fn for fn in os.listdir(os.path.join(submRoot, "phonetic/dev-clean")) if os.path.isfile(os.path.join(submRoot, "phonetic/dev-clean", fn)) and fn.split('.')[0] in sDevClean]
submFilesDevCleanBad = [fn for fn in os.listdir(os.path.join(submRoot, "phonetic/dev-clean")) if os.path.isfile(os.path.join(submRoot, "phonetic/dev-clean", fn)) and not fn.split('.')[0] in sDevClean]
submFilesDevOtherOk = [fn for fn in os.listdir(os.path.join(submRoot, "phonetic/dev-other")) if os.path.isfile(os.path.join(submRoot, "phonetic/dev-other", fn)) and fn.split('.')[0] in sDevOther]
submFilesDevOtherBad = [fn for fn in os.listdir(os.path.join(submRoot, "phonetic/dev-other")) if os.path.isfile(os.path.join(submRoot, "phonetic/dev-other", fn)) and not fn.split('.')[0] in sDevOther]

print(len(submFilesDevCleanOk), len(submFilesDevCleanBad))
print(len(submFilesDevOtherOk), len(submFilesDevOtherBad))

print(f'Bad files: {len(submFilesDevCleanBad) + len(submFilesDevOtherBad)}, ok files: {len(submFilesDevCleanOk)}, {len(submFilesDevOtherOk)}')

os.makedirs(newSubmRoot + "/phonetic")
os.makedirs(newSubmRoot + "/phonetic/dev-clean")
os.makedirs(newSubmRoot + "/phonetic/dev-other")
#if not os.path.exists(shorteningRoot + "/lexical"):
os.makedirs(newSubmRoot + "/lexical")
#if not os.path.exists(shorteningRoot + "/syntactic"):
os.makedirs(newSubmRoot + "/syntactic")
#if not os.path.exists(shorteningRoot + "/semantic"):
os.makedirs(newSubmRoot + "/semantic")
#print(metadataFileToCpy)
copyfile(metadataFileToCpy, newSubmRoot + "/meta.yaml")

i = 0
for fn in submFilesDevCleanOk:
    copyfile(os.path.join(submRoot, "phonetic/dev-clean", fn), os.path.join(newSubmRoot, "phonetic/dev-clean", fn))
    i += 1
    if i % 100 == 0:
        print(f'{i} dev-clean files copied')

i = 0
for fn in submFilesDevOtherOk:
    copyfile(os.path.join(submRoot, "phonetic/dev-other", fn), os.path.join(newSubmRoot, "phonetic/dev-other", fn))
    i += 1
    if i % 100 == 0:
        print(f'{i} dev-other files copied')

# badFiles = 0
# for fn in dirFiles:
#     if not fn.endswith(".wav"):
#         continue
#     fbase = fn.split('.')[0]
#     if fbase not in s:
#         badFiles += 1

# print(f"files outside of their DS used for ABX: {badFiles}")