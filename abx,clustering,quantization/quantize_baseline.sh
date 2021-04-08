



LibriSpeechPath=$1
LibriSpeechFormat=$2  # e.g. flac; without dot
evalDSpath=$3
evalDSformat=$4  # e.g. wav; without dot
saveThingsPath=$5

# this is for already done baseline euclidean quantization and euclidean assignments
normalizationSettingAssignment=dontnormalize  # euclidean-closest cluster assignment

# this is for baseline checkpoint
nullspaceTypeOfCheckpoint=nonullspace

# baseline checkpoint with baseline clustering data
baseModelCheckpoint=$6  

PREVIFS=$IFS

# with batch set, context is not kept between the audio files
# [!] in case of GPU overflow, change batch size below - after 2nd comma in each line below; for evalDS files in batch are smaller
setsToProcess=\
"${LibriSpeechPath}/train-clean-100,${saveThingsPath}/baseline_quantizations/LibriSpeech/train-clean-100,8,${LibriSpeechFormat} \
${LibriSpeechPath}/train-full-960,${saveThingsPath}/baseline_quantizations/LibriSpeech/train-full-960,8,${LibriSpeechFormat} \
${evalDSpath}/lexical/dev,${saveThingsPath}/baseline_quantizations/evalDS/lexical/dev,8,${evalDSformat} \
${evalDSpath}/lexical/test,${saveThingsPath}/baseline_quantizations/evalDS/lexical/test,8,${evalDSformat} \
${evalDSpath}/semantic/dev/librispeech,${saveThingsPath}/baseline_quantizations/evalDS/semantic/dev/librispeech,8,${evalDSformat} \
${evalDSpath}/semantic/dev/synthetic,${saveThingsPath}/baseline_quantizations/evalDS/semantic/dev/synthetic,8,${evalDSformat} \
${evalDSpath}/semantic/test/librispeech,${saveThingsPath}/baseline_quantizations/evalDS/semantic/test/librispeech,8,${evalDSformat} \
${evalDSpath}/semantic/test/synthetic,${saveThingsPath}/baseline_quantizations/evalDS/semantic/test/synthetic,8,${evalDSformat} \
${evalDSpath}/syntactic/dev,${saveThingsPath}/baseline_quantizations/evalDS/syntactic/dev,8,${evalDSformat} \
${evalDSpath}/syntactic/test,${saveThingsPath}/baseline_quantizations/evalDS/syntactic/test,8,${evalDSformat}"

IFS=' '
for i in $setsToProcess
do
    IFS=','
    set -- $i
    echo "quantizing from ${1} to ${2}, batch ${3}, audio format ${4}"

    mkdir -p $2

    ./quantize.sh \
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