

zerospeech2021_baseline_dir=$1
LibriSpeechDir=$2
ZSdatasetDir=$3
saveDir=$4
audioFormat=$5
condaPath=$6
ZSbaselineEnv=$7
ZSenv=$8


mkdir -p $saveDir/reproduce_baseline_ABX_submission
mkdir -p $saveDir/reproduce_baseline_ABX_submission/phonetic
mkdir -p $saveDir/reproduce_baseline_ABX_submission/lexical
mkdir -p $saveDir/reproduce_baseline_ABX_submission/semantic
mkdir -p $saveDir/reproduce_baseline_ABX_submission/syntactic

cat > $saveDir/reproduce_baseline_ABX_submission/meta.yaml << EOF
author: placeholder
affiliation: placeholder
description: placeholder
open_source: true
train_set: placeholder
gpu_budget: placeholder
parameters:
  phonetic:
    metric: cosine
    frame_shift: 0.01
EOF

source $condaPath/etc/profile.d/conda.sh
savedEnv=$(conda info | sed -n 's/\( \)*active environment : //p')
echo SAVED_ENV: $savedEnv

conda activate $ZSbaselineEnv

python $zerospeech2021_baseline_dir/scripts/build_CPC_features.py \
$zerospeech2021_baseline_dir/checkpoints/CPC-big-kmeans50/cpc_ll6k/checkpoint_32.pt \
$LibriSpeechDir/dev-clean \
$saveDir/reproduce_baseline_ABX_submission/phonetic/dev-clean \
--file_extension $audioFormat --gru_level 2

python $zerospeech2021_baseline_dir/scripts/build_CPC_features.py \
$zerospeech2021_baseline_dir/checkpoints/CPC-big-kmeans50/cpc_ll6k/checkpoint_32.pt \
$LibriSpeechDir/dev-other \
$saveDir/reproduce_baseline_ABX_submission/phonetic/dev-other \
--file_extension $audioFormat --gru_level 2

python $zerospeech2021_baseline_dir/scripts/build_CPC_features.py \
$zerospeech2021_baseline_dir/checkpoints/CPC-big-kmeans50/cpc_ll6k/checkpoint_32.pt \
$LibriSpeechDir/test-clean \
$saveDir/reproduce_baseline_ABX_submission/phonetic/test-clean \
--file_extension $audioFormat --gru_level 2

python $zerospeech2021_baseline_dir/scripts/build_CPC_features.py \
$zerospeech2021_baseline_dir/checkpoints/CPC-big-kmeans50/cpc_ll6k/checkpoint_32.pt \
$LibriSpeechDir/test-other \
$saveDir/reproduce_baseline_ABX_submission/phonetic/test-other \
--file_extension $audioFormat --gru_level 2


python clean_phonetic_features.py $ZSdatasetDir/phonetic/dev-clean $saveDir/reproduce_baseline_ABX_submission/phonetic/dev-clean

python clean_phonetic_features.py $ZSdatasetDir/phonetic/dev-other $saveDir/reproduce_baseline_ABX_submission/phonetic/dev-other

python clean_phonetic_features.py $ZSdatasetDir/phonetic/test-clean $saveDir/reproduce_baseline_ABX_submission/phonetic/test-clean

python clean_phonetic_features.py $ZSdatasetDir/phonetic/test-other $saveDir/reproduce_baseline_ABX_submission/phonetic/test-other


conda activate $ZSenv

zerospeech2021-evaluate --force-cpu --no-lexical --no-syntactic --no-semantic -j 20 -o \
$saveDir/reproduce_baseline_ABX_submission_eval $ZSdatasetDir $saveDir/reproduce_baseline_ABX_submission/

if [ $savedEnv!=None ]; then
  conda activate $savedEnv
fi