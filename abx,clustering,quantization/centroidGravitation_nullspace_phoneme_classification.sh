
LibriSpeechDir=$1
LStrainLabelsFile=$2  # zd/LibriSpeech/labels_split/train_split_100.txt
LStestLabelsFile=$3  #zd/LibriSpeech/labels_split/test_split_100.txt
noClusteringNoNullspaceCheckpointPath=$4  # $zd/checkpoints/CPC-big-kmeans50/cpc_ll6k/checkpoint_32.pt
clusteringNullspaceCheckpointPath=$5  #  $cpcClustDir/checkpoints/clustering_CPC_big_kmeans50_nullspace_64/clustering_CPC_big_kmeans50_nullspace_64.pt
nullspaceSpeakersFactorizedCheckpointPath=$6  # $nullspaceDir/linear_separability/cpc/gru_level2/cpc_official_speakers_factorized_64/checkpoint_9.pt
phoneAlignmentsFile=$7  # $zd/LibriSpeech/alignments2/converted_aligned_phones.txt
saveDir=$8


for deg in 0 0.2 0.3 0.4 0.5 0.6 0.7
do
    echo $deg
    mkdir $saveDir/phoneme_classif_null_${deg}/
    python cpc/eval/linear_separability.py $LibriSpeechDir/train-clean-100/ \
    $LStrainLabelsFile \
    $LStestLabelsFile \
    $noClusteringNoNullspaceCheckpointPath \
    --centerpushFile $clusteringNullspaceCheckpointPath \
    --centerpushDeg $deg \
    --pathCheckpoint $saveDir/phoneme_classif_null_${deg}/ \
    --mode phonemes_nullspace --max_size_loaded 40000000 --n_process_loader 2 \
    --model cpc --pathPhone $phoneAlignmentsFile \
    --path_speakers_factorized $nullspaceSpeakersFactorizedCheckpointPath \
    --dim_inter 64 --gru_level 2 --batchSizeGPU 32 | tee $saveDir/phoneme_classif_null_${deg}/log.txt
done

