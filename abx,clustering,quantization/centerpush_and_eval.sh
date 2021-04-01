
# cos clustering, cos assignment, nullspace


clustersCheckpoint=$1  # /pio/gluster/i283340/cosine_quant/nullspace64trainedLStrain-clean-100/trained50clusters.pt

saveSubmissionRoot=$2  # /pio/gluster/i283340/subms_nullClean/subm_nullspaceCosClustCosAssign
# saves files with modified embeddings under there, more in the next param below; those files can be later copied to actual submission

inOutSubsetsDescr=$3  # this is in the following format:
# a1{}b2:a2{}b2 etc., where:
# a = subset_relative_path_to_save_under_saveSubmissionRoot/phonetic  (e.g. for dev-clean will save embeddings under saveSubmissionRoot/phonetic/dev-clean)
# b = path_to_directory_with_saved_embeddings_for_this_subset  (those also need to be in the .txt format)
# example:  dev-clean{}../null-features/phonetic/dev-clean:dev-other{}../null-features/phonetic/dev-other

metaFileToUse=$4  # /pio/gluster/i283340/subms_null/subm_nullspaceCosClustCosAssign_deg0-6/meta.yaml
# this is copied to created submission dir in order to automatically run phonetic evaluation afterwards

phoneticEvalDSpath=$5  # /pio/scratch/1/i283340/MGR/zs/quantization/ls-ds-dev/

batchSize=32  # integer, e.g. 32
processesToUse=20  # integer (e.g. 20) or cuda

degList=$6  # list of push degrees as batch list, e.g. "0.2 0.3 0.4"

closestClusterChoice=$7  # cosineclosest or euclideanclosest 
if [ $closestClusterChoice==cosineclosest ]; then
    closestClusterCosine=True
elif [ $closestClusterChoice==euclideanclosest ]; then
    closestClusterCosine=False
else
    echo "invalid closestClusterChoice config choice; has to be cosineclosest or euclideanclosest"
    exit 1
fi

normalizeForPushChoice=$8  # normalizeforpush or dontnormalizeforpush  ; normalizing aims to approximate pushing part of cosine and not euclidean distance
if [ $normalizeForPushChoice==normalizeforpush ]; then
    normalizeForPush=True
elif [ $normalizeForPushChoice==dontnormalizeforpush ]; then
    normalizeForPush=False
else
    echo "invalid normalizeForPushChoice config choice; has to be normalizeforpush or dontnormalizeforpush"
    exit 1
fi

for deg in $degList ; do
    python closestpush.py \
    $clustersCheckpoint \
    $saveSubmissionRoot \
    $inOutSubsetsDescr \
    $metaFileToUse \
    $batchSize \
    $processesToUse \
    $deg \
    $closestClusterCosine \
    $normalizeForPush
done

./evalmore.sh ${saveSubmissionRoot}_deg ${saveSubmissionRoot}_ABXeval_deg $degList $phoneticEvalDSpath ${saveSubmissionRoot}_ABXeval_output.txt

