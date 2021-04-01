
pathToDS=$1  # /pio/data/zerospeech2021/LibriSpeech/train-clean-100/
checkpointWithoutClusters=$2  # /pio/scratch/1/i273233/linear_separability/cpc/gru_level2/cpc_official_phonemes_nullspace_64/checkpoint_9.pt
clusteringCheckpointSavePath=$3  #/pio/gluster/i283340/cosine_quant/nullspace64trainedLStrain-clean-100/trained50clusters.pt

nullspaceTypeOfCheckpoint=$4  # nullspace or nonullspace 
if [ $nullspaceTypeOfCheckpoint==nullspace ]; then
    nullspaceFlag=--nullspace
elif [ $nullspaceTypeOfCheckpoint==nonullspace ]; then
    nullspaceFlag=
else
    echo "invalid nullspace config choice; has to be nullspace or nonullspace"
    exit 1
fi

normalizationSetting=$5  # normalize or dontnormalize 
if [ $normalizationSetting==normalize ]; then
    normalizationFlag=--norm_vec_len
elif [ $normalizationSetting==dontnormalize ]; then
    normalizationFlag=
else
    echo "invalid normalize config choice; has to be normalize or dontnormalize"
    exit 1
fi

maxIter=$6  # 150
formatExtension=$7  # without dot

python CPC_audio/cpc/criterion/clustering/clustering_script.py \
--pathDB $pathToDS \
--recursionLevel 1 --nClusters 50 --MAX_ITER $maxIter --level_gru 2 --save --load \
--batchSizeGPU 500 --max_size_loaded 40000000 --n_process_loader 2 \
$nullspaceFlag $normalizationFlag \
$checkpointWithoutClusters \
$clusteringCheckpointSavePath \
--extension ".${formatExtension}"


# TODO scripts using another scripts so that it's simpler (e.g. script using cluster-making script and then doing quantization and setting part of the args used in between)