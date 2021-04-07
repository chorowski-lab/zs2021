

clusteringCheckpoint=$1  # /pio/gluster/i283340/cosine_quant/nullspace64trainedLStrain-clean-100/trained50clusters.pt
DSpath=$2  # /pio/data/zerospeech2021/dataset/lexical/dev
savePath=$3  # /pio/gluster/i283340/cosine_clusters_cosine_assignments/lexical/dev/
bsize=$4
fileExt=$5

nullspaceTypeOfCheckpoint=$6  # nullspace or nonullspace echo $nullspaceTypeOfCheckpoint
if [ $nullspaceTypeOfCheckpoint == "nullspace" ]; then
    nullspaceFlag=--nullspace
elif [ $nullspaceTypeOfCheckpoint == "nonullspace" ]; then
    nullspaceFlag=
else
    echo "invalid nullspace config choice; has to be nullspace or nonullspace"
    exit 1
fi

normalizationSetting=$7  # normalize or dontnormalize 
if [ $normalizationSetting == "normalize" ]; then
    normalizationFlag=--norm_vec_len
elif [ $normalizationSetting == "dontnormalize" ]; then
    normalizationFlag=
else
    echo "invalid normalize config choice; has to be normalize or dontnormalize"
    exit 1
fi

python CPC_audio/scripts/quantize_audio.py \
$clusteringCheckpoint \
$DSpath \
$savePath \
--batch_size $bsize $nullspaceFlag $normalizationFlag --file_extension $fileExt
