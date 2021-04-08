
LibriSpeechDir=$1
LStrainLabelsFile=$2
LStestLabelsFile=$3
noClusteringNoNullspaceCheckpointPath=$4
clusteringNullspaceCheckpointPath=$5
nullspaceSpeakersFactorizedCheckpointPath=$6
phoneAlignmentsFile=$7
saveDir=$8


for deg in 0 0.2 0.3 0.4 0.5 0.6 0.7
do
    echo $deg
    mkdir -p $saveDir/centroid_gravitation_phoneme_classif/null_${deg}/
    python CPC_audio/cpc/eval/linear_separability.py $LibriSpeechDir/train-clean-100/ \
    $LStrainLabelsFile \
    $LStestLabelsFile \
    $noClusteringNoNullspaceCheckpointPath \
    --centerpushFile $clusteringNullspaceCheckpointPath \
    --centerpushDeg $deg \
    --pathCheckpoint $saveDir/centroid_gravitation_phoneme_classif/null_${deg}/ \
    --mode phonemes_nullspace --max_size_loaded 40000000 --n_process_loader 2 \
    --model cpc --pathPhone $phoneAlignmentsFile \
    --path_speakers_factorized $nullspaceSpeakersFactorizedCheckpointPath \
    --dim_inter 64 --gru_level 2 --batchSizeGPU 32 | tee $saveDir/centroid_gravitation_phoneme_classif/null_${deg}/log.txt
    # THIS ASSUMES dim_inter 64
done


