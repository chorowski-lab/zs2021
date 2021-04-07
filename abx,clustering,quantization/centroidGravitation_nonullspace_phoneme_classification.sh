

LibriSpeechDir=$1
LStrainLabelsFile=$2  # zd/LibriSpeech/labels_split/train_split_100.txt
LStestLabelsFile=$3  #zd/LibriSpeech/labels_split/test_split_100.txt
noClusteringNoNullspaceCheckpointPath=$4  # $zd/checkpoints/CPC-big-kmeans50/cpc_ll6k/checkpoint_32.pt
clusteringNoNullspaceCheckpointPath=$5  # $zd/checkpoints/CPC-big-kmeans50/clustering_kmeans50/clustering_CPC_big_kmeans50.pt
phoneAlignmentsFile=$6  # $zd/LibriSpeech/alignments2/converted_aligned_phones.txt
saveDir=$7

for deg in 0 0.2 0.3 0.4 0.5 0.6 0.7
do
    echo $deg
    mkdir $saveDir/phoneme_classif_nonull_${deg}/
    python CPC_audio/cpc/eval/linear_separability.py $LibriSpeechDir/train-clean-100/ \
    $LStrainLabelsFile \
    $LStestLabelsFile \
    $noClusteringNoNullspaceCheckpointPath \
    --centerpushFile $clusteringNoNullspaceCheckpointPath \
    --centerpushDeg $deg \
    --pathCheckpoint $saveDir/phoneme_classif_nonull_${deg}/ \
    --mode phonemes --max_size_loaded 40000000 --n_process_loader 2 \
    --model cpc --pathPhone $phoneAlignmentsFile \
    --gru_level 2 --batchSizeGPU 32 | tee $saveDir/phoneme_classif_nonull_${deg}/log.txt
done