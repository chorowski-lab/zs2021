

embeddingsRoot=$1  # assuming will contain dev/test-clean/other subdirs underneath

clustersCheckpoint=$2  # /pio/gluster/i283340/cosine_quant/nullspace64trainedLStrain-clean-100/trained50clusters.pt

saveSubmissionsRoot=$3

phoneticEvalDSpath=$4

degList=$5  # list of push degrees as batch list, e.g. "0.2 0.3 0.4"

closestClusterChoice=$6  # cosineclosest or euclideanclosest 

normalizeForPushChoice=$7  # normalizeforpush or dontnormalizeforpush  ; normalizing aims to approximate pushing part of cosine and not euclidean distance

mkdir -p $saveSubmissionsRoot

cat > $saveSubmissionsRoot/meta.yaml << EOF
author: placeholder
affiliation: placeholder
description: placeholder
open_source: true
train_set: placeholder
gpu_budget: placeholder
parameters:
  phonetic:
    metric: cosine
    frame_shift: 0.01
EOF

centerpush_and_eval.sh \
$clustersCheckpoint \
"${saveSubmissionsRoot}_${closestClusterChoice}_${normalizeForPushChoice}" \
"dev-clean{}${embeddingsRoot}/dev-clean:dev-other{}${embeddingsRoot}/dev-other:test-clean{}${embeddingsRoot}/test-clean:test-other{}${embeddingsRoot}/test-other" \
$saveSubmissionsRoot/meta.yaml \
$phoneticEvalDSpath \
$degList \
$closestClusterChoice \
$normalizeForPushChoice