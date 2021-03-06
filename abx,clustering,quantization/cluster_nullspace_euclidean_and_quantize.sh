



LibriSpeechPath=$1
LibriSpeechFormat=$2  # e.g. flac; without dot
evalDSpath=$3
evalDSformat=$4  # e.g. wav; without dot
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

# with batch set, context is not kept between the audio files
# [!] in case of GPU overflow, change batch size below - after 2nd comma in each line below; for evalDS files in batch are smaller
setsToProcess=\
"${LibriSpeechPath}/train-clean-100,${saveThingsPath}/nullspace_euclidean_euclidean_quantizations/LibriSpeech/train-clean-100,32,${LibriSpeechFormat} \
${LibriSpeechPath}/train-full-960,${saveThingsPath}/nullspace_euclidean_euclidean_quantizations/LibriSpeech/train-full-960,32,${LibriSpeechFormat} \
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