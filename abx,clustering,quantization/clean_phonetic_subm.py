


import sys
import os


dsRoot = sys.argv[1]
submRoot = sys.argv[2]


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

i = 0
for fn in submFilesDevCleanBad:  
    os.remove(os.path.join(submRoot, "phonetic/dev-clean", fn))
    i += 1
    if i % 100 == 0:
        print(f'{i} dev-clean files copied')

i = 0
for fn in submFilesDevOtherBad:  
    os.remove(os.path.join(submRoot, "phonetic/dev-other", fn))
    i += 1
    if i % 100 == 0:
        print(f'{i} dev-other files copied')

