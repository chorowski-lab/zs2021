



LibriSpeechPath=$1  # /pio/data/zerospeech2021/LibriSpeech
LibriSpeechFormat=$2  # flac; without dot
evalDSpath=$3   # /pio/data/zerospeech2021/dataset/
evalDSformat=$4  # wav; without dot
saveThingsPath=$5

# this is for euclidean quantization and euclidean assignments
normalizationSettingClustering=dontnormalize  # euclidean clustering
normalizationSettingAssignment=dontnormalize  # euclidean-closest cluster assignment

# this is for nullspace-based checkpoint
nullspaceTypeOfCheckpoint=nullspace

# base checkpoint without clustering data, needs to be consistent with nullspaceTypeOfCheckpoint
baseModelCheckpoint=$6  

./cluster_embeddings.sh \
${LibriSpeechPath}/train-clean-100/ \
$baseModelCheckpoint \
${saveThingsPath}/trained_nullspace_euclidean_kmeans/kmeans50checkpoint.pt \
$nullspaceTypeOfCheckpoint \
$normalizationSettingClustering \
150 \
$LibriSpeechFormat

PREVIFS=$IFS

# we didn't use option to discard context between files when processing evaluation set with e.g. word per file
# hoping that maybe it will still be better than cold start
setsToProcess=\
"${LibriSpeechPath}/LibriSpeech/train-clean-100,${saveThingsPath}/nullspace_euclidean_euclidean_quantizations/LibriSpeech/train-clean-100,32,${LibriSpeechFormat} \
${LibriSpeechPath}/LibriSpeech/train-full-960,${saveThingsPath}/nullspace_euclidean_euclidean_quantizations/LibriSpeech/train-full-960,32,${LibriSpeechFormat} \
${evalDSpath}/lexical/dev,${saveThingsPath}/nullspace_euclidean_euclidean_quantizations/evalDS/lexical/dev,128,${evalDSformat} \
${evalDSpath}/lexical/test,${saveThingsPath}/nullspace_euclidean_euclidean_quantizations/evalDS/lexical/test,128,${evalDSformat} \
${evalDSpath}/semantic/dev/librispeech,${saveThingsPath}/nullspace_euclidean_euclidean_quantizations/evalDS/semantic/dev/librispeech,128,${evalDSformat} \
${evalDSpath}/semantic/dev/synthetic,${saveThingsPath}/nullspace_euclidean_euclidean_quantizations/evalDS/semantic/dev/synthetic,128,${evalDSformat} \
${evalDSpath}/semantic/test/librispeech,${saveThingsPath}/nullspace_euclidean_euclidean_quantizations/evalDS/semantic/test/librispeech,128,${evalDSformat} \
${evalDSpath}/semantic/test/synthetic,${saveThingsPath}/nullspace_euclidean_euclidean_quantizations/evalDS/semantic/test/synthetic,128,${evalDSformat} \
${evalDSpath}/syntactic/dev,${saveThingsPath}/nullspace_euclidean_euclidean_quantizations/evalDS/syntactic/dev,128,${evalDSformat} \
${evalDSpath}/syntactic/test,${saveThingsPath}/nullspace_euclidean_euclidean_quantizations/evalDS/syntactic/test,128,${evalDSformat}"

IFS=' '
for i in $setsToProcess
do
    IFS=','
    set -- $i
    echo "quantizing from ${1} to ${2}, batch ${3}, audio format ${4}"

    mkdir -p $2

    ./quantize.sh \
    ${saveThingsPath}/trained_nullspace_euclidean_kmeans/kmeans50checkpoint.pt \
    $1 \
    $2 \
    $3 \
    $4 \
    $nullspaceTypeOfCheckpoint \
    $normalizationSettingAssignment

    IFS=' '
done

IFS=$PREVIFS