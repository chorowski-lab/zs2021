



LibriSpeechPath=$1  # /pio/data/zerospeech2021/LibriSpeech
LibriSpeechFormat=$2  # flac; without dot
evalDSpath=$3   # /pio/data/zerospeech2021/dataset/
evalDSformat=$4  # wav; without dot
saveThingsPath=$5

# this is for already done baseline euclidean quantization and euclidean assignments
normalizationSettingAssignment=nonormalize  # euclidean-closest cluster assignment

# this is for baseline checkpoint
nullspaceTypeOfCheckpoint=nonullspace

# baseline checkpoint with baseline clustering data
baseModelCheckpoint=$6  

PREVIFS=$IFS

# we didn't use option to discard context between files when processing evaluation set with e.g. word per file
# hoping that maybe it will still be better than cold start
setsToProcess=\
"${LibriSpeechPath}/LibriSpeech/train-clean-100,${saveThingsPath}/baseline_quantizations/LibriSpeech/train-clean-100,32,${LibriSpeechFormat} \
${LibriSpeechPath}/LibriSpeech/train-full-960,${saveThingsPath}/baseline_quantizations/LibriSpeech/train-full-960,32,${LibriSpeechFormat} \
${evalDSpath}/lexical/dev,${saveThingsPath}/baseline_quantizations/evalDS/lexical/dev,128,${evalDSformat} \
${evalDSpath}/lexical/test,${saveThingsPath}/baseline_quantizations/evalDS/lexical/test,128,${evalDSformat} \
${evalDSpath}/semantic/dev/librispeech,${saveThingsPath}/baseline_quantizations/evalDS/semantic/dev/librispeech,128,${evalDSformat} \
${evalDSpath}/semantic/dev/synthetic,${saveThingsPath}/baseline_quantizations/evalDS/semantic/dev/synthetic,128,${evalDSformat} \
${evalDSpath}/semantic/test/librispeech,${saveThingsPath}/baseline_quantizations/evalDS/semantic/test/librispeech,128,${evalDSformat} \
${evalDSpath}/semantic/test/synthetic,${saveThingsPath}/baseline_quantizations/evalDS/semantic/test/synthetic,128,${evalDSformat} \
${evalDSpath}/syntactic/dev,${saveThingsPath}/baseline_quantizations/evalDS/syntactic/dev,128,${evalDSformat} \
${evalDSpath}/syntactic/test,${saveThingsPath}/baseline_quantizations/evalDS/syntactic/test,128,${evalDSformat}"

IFS=' '
for i in $setsToProcess
do
    IFS=','
    set -- $i
    echo "quantizing from ${1} to ${2}, batch ${3}, audio format ${4}"

    mkdir -p $2

    quantize.sh \
    $baseModelCheckpoint \
    $1 \
    $2 \
    $3 \
    $4 \
    $nullspaceTypeOfCheckpoint \
    $normalizationSettingAssignment

    IFS=' '
done

IFS=$PREVIFS