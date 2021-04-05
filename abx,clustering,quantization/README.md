

In the `CPC_audio` folder there is a snapshot of our <https://github.com/chorowski-lab/CPC_audio> repo (`ZS_snapshot` branch), which also uses some code from other branches/repos than main one of its parent repo (see top of its readme under `CPC_audio` folder here).

The code in this folder has been used for:
  - producing nullspace-based embeddings
  - performing Euclidean and cosine k-means clustering in the nullspace
  - producing embeddings' quantizations for tasks other than ABX
  - using 'centroid-gravitation'
    
Additionally, cluster centers distance matrix to use for sWUGGY task was made (see `dist_matrix_from_clusters.py`).


## How to reproduce the results:

Names written as variables in capital letters are to be changed to user-specific when executing this


### General preparation

1. Install conda; $CONDA_PATH will denote install path
2. Download LibriSpeech dataset (in .flac format, otherwise need to change formats in later commands) and place under $LIBRISPEECH_DATASET_PATH
3. Download ZeroSpeech2021 evaluation dataset (in .wav format) and place under $ZEROSPEECH_DATASET_PATH
4. Setup `zerospeech2021` repo with its conda env (<https://github.com/bootphon/zerospeech2021>) according to its readme; $ZEROSPEECH_EVAL_ENVIRONMENT will denote env name (zerospeech2021 by default)
5. Setup `zerospeech2021_baseline` repo with its conda env (<https://github.com/bootphon/zerospeech2021_baseline>) according to its readme (including downloading checkpoints) under $ZEROSPEECH2021_BASELINE_PATH; $ZEROSPEECH_BASELINE_ENVIRONMENT will denote env name (zerospeech2021_baseline by default)
6. Install `CPC_audio` repo snapshot provided in `CPC_audio` directory here with its conda env according to its readme; $CPC_ENVIRONMENT denotes the env name (cpc_audio by default; please note you may need to change this before installation if you already have existing env with same name)
7. Make a folder for saving results - $SAVE_DIR
8. Create/download LibriSpeech train-clean-100 split into train and test sets saved in .txt files where each line contains name of file belonging in this split part (e.g. 7780-274562-0073 etc.); $LIBRISPEECH_TRAIN_CLEAN_100_TRAIN_SPLIT_FILE_PATH and $LIBRISPEECH_TRAIN_CLEAN_100_TRAIN_SPLIT_FILE_PATH will denote locations of those files
9. Download phoneme alignments in correct format, which is linked in `CPC_audio` repo readme and place it under $PHONEME_ALIGNMENTS_FILE


### Reproducing baseline ABX with workflow described in the submission paper:

We were only able to reproduce baseline results with following workflow:
 - compute representations for the whole LibriSpeech-dev/test dataset
 - remove extra files
 - compute other things on top of those
which achieved better results in comparison to computing features for datasets used for ZeroSpeech phonetic metric evaluation. This can perhaps be because audio data in LibriSpeech tends to be consecutive and removing parts of it (some files, as in ZeroSpeech ABX-evaluation dataset) may harm autoregressive context (as `zerospeech2021_baseline/scripts/build_CPC_features.py` script we used for building features keeps autoregressive context between files as default, so removing some audio files was perhaps making high-level features out-of-date (by some files))

1. Run (this will source conda and activate needed envs):
```bash
./reproduce_baseline_ABX.sh \
$ZEROSPEECH2021_BASELINE_PATH $LIBRISPEECH_DATASET_PATH $ZEROSPEECH_DATASET_PATH \
$SAVE_DIR flac $CONDA_PATH $ZEROSPEECH_BASELINE_ENVIRONMENT $ZEROSPEECH_EVAL_ENVIRONMENT
```

This will leave produced embeddings under `$SAVE_DIR/reproduce_baseline_ABX_submission/phonetic`.


### Nullspace experiments

1. Activate `CPC_audio` conda env
2. Run `finetune_nullspace.sh` from `CPC_audio` directory:

```
Usage: ./finetune_nullspace.sh
        -d LIBRISPEECH_DATASET_PATH/train-clean-100 
        -t LIBRISPEECH_TRAIN_CLEAN_100_TRAIN_SPLIT_FILE_PATH
        -v LIBRISPEECH_TRAIN_CLEAN_100_TEST_SPLIT_FILE_PATH
        -c BASELINE_NO_CLUSTERING_CHECKPOINT_PATH
        -o SAVE_DIR
        -n DIM_INBETWEEN (Dimension of nullspace will be DIM_EMBEDDING - DIM_INBETWEEN)
OPTIONAL ARGS:
        -s FROM_STEP (From which step do you want to start. Order: speakers_factorized [default] -> phonemes_nullspace -> speakers_nullspace)
        -p PHONEME_ALIGNMENTS_FILE (Path to the file containing phonemes for the entire dataset. You don't need it if you start from speakers_nullspace)
        -f AUDIO_FORMAT (audio files format in LibriSpeech dataset (without a dot))
```
In order to reproduce our experiment from the paper, run the following:

```bash
for i in 256 320 416 448 464
do
    ./finetune_nullspace.sh -d $LIBRISPEECH_DATASET_PATH/train-clean-100 -t $LIBRISPEECH_TRAIN_CLEAN_100_TRAIN_SPLIT_FILE_PATH -v $LIBRISPEECH_TRAIN_CLEAN_100_TEST_SPLIT_FILE_PATH -c $BASELINE_NO_CLUSTERING_CHECKPOINT_PATH -o $SAVE_DIR/nullspaces/$i -n $(expr 512 - $i) -p $PHONEME_ALIGNMENTS_FILE -f flac
done
```

This will leave its results in subfolder(s) under $SAVE_DIR


### Evaluating ABX for nullspace

1. Create the flattened version of LibriSpeech dev/test dataset. Once you have done this you do not have to do it anymore:
```bash
for directory in dev-clean dev-other test-clean test-other
do
  mkdir -p $LIBRISPEECH_FLATTENED_DATASET_PATH/phonetic/$directory
  cp $LIBRISPEECH_DATASET_PATH/$directory/*/*/*.flac $LIBRISPEECH_FLATTENED_DATASET_PATH/phonetic/$directory
done

for directory in dev-clean dev-other
do
  cp $ZEROSPEECH_DATASET_PATH/phonetic/$directory/$directory.item $LIBRISPEECH_FLATTENED_DATASET_PATH/phonetic/$directory
done
```
2. Complete "Nullspace experiments" section
3. Run `scripts/eval_abx.sh` from `CPC_audio` directory (it activates needed envs passed as arguments):

```
Usage: scripts/eval_abx.sh
        -d DATASET_PATH (Either ZEROSPEECH_DATASET_PATH or LIBRISPEECH_FLATTENED_DATASET_PATH)
        -r ZEROSPEECH_DATASET_PATH
        -c CHECKPOINT_PATH
        -o SAVE_DIR
OPTIONAL ARGS:
        -n (Provide this flag if you want to load a model with nullspace)
        -a CONDA_PATH
        -e CPC_ENVIRONMENT
        -z ZEROSPEECH_EVAL_ENVIRONMENT (The conda environment where the zerospeech2021-evaluate is installed)
        -t (Do not compute embeddings for test set)
        -f AUDIO_FORMAT (audio files format in LibriSpeech dataset (without a dot))
```
In order to reproduce ABX error rates for a CPC + nullspace, run the following (Note that LIBRISPEECH_FLATTENED_DATASET_PATH refers to the dateset created earlier and not to the path where LibriSpeech is located):

```bash
#  --> provide chosen nullspace model that first needs to be produced like in "nullspace experiments" section as $CHECKPOINT_PATH
#      those checkpoints should be under $SAVE_DIR/nullspaces/$(512-DIM_NULLSPACE)/(speakers_factorized OR phonemes_nullspace OR speakers_nullspace)_$(512-DIM_NULLSPACE)/checkpoint9.pt
#      after "nullspace experiments" section; note that (speakers_factorized OR phonemes_nullspace OR speakers_nullspace) refers to different rows in ABX error rates table in the paper

scripts/eval_abx.sh -d $ZEROSPEECH_DATASET_PATH -r $ZEROSPEECH_DATASET_PATH -c $CHECKPOINT_PATH -o $SAVE_DIR/nullspace_abx_original -n -a $CONDA_PATH -e $CPC_ENVIRONMENT -z $ZEROSPEECH_EVAL_ENVIRONMENT -f wav

scripts/eval_abx.sh -d $LIBRISPEECH_FLATTENED_DATASET_PATH -r $ZEROSPEECH_DATASET_PATH -c $CHECKPOINT_PATH -o $SAVE_DIR/nullspace_abx_librispeech -n -a $CONDA_PATH -e $CPC_ENVIRONMENT -z $ZEROSPEECH_EVAL_ENVIRONMENT -f flac

```

This will leave its results in subfolder(s) under $SAVE_DIR


### Performing k-means clustering on nullspace embeddings and producing quantizations:

1. Complete "Nullspace experiments" section
2. Activate `CPC_audio` env
3. For cosine k-means clustering and cosine-closest assignment quantizations run:
  ```bash
  #  --> provide chosen nullspace model that first needs to be produced like in "nullspace experiments" section as $NULLSPACE_MODEL_NO_CLUSTERING_CHECKPOINT_PATH
  #      those checkpoints should be under $SAVE_DIR/nullspaces/$(512-DIM_NULLSPACE)/phonemes_nullspace_$(512-DIM_NULLSPACE)/checkpoint9.pt

  ./cluster_nullspace_cosine_and_quantize.sh \
  $LIBRISPEECH_DATASET_PATH flac $ZEROSPEECH_DATASET_PATH wav \
  $SAVE_DIR $NULLSPACE_MODEL_NO_CLUSTERING_CHECKPOINT_PATH
  ```
  For euclidean k-means clustering and euclidean-closest assignment quantizations run:
  ```bash
  #  --> provide chosen nullspace model that first needs to be produced like in "nullspace experiments" section as $NULLSPACE_MODEL_NO_CLUSTERING_CHECKPOINT_PATH
  #      those checkpoints should be under $SAVE_DIR/nullspaces/$(512-DIM_NULLSPACE)/phonemes_nullspace_$(512-DIM_NULLSPACE)/checkpoint9.pt

  ./cluster_nullspace_euclidean_and_quantize.sh \
  $LIBRISPEECH_DATASET_PATH flac $ZEROSPEECH_DATASET_PATH wav \
  $SAVE_DIR $NULLSPACE_MODEL_NO_CLUSTERING_CHECKPOINT_PATH
  ```

This will leave its results in subfolder(s) under $SAVE_DIR


### Producing baseline quantizations

1. Activate `CPC_audio` env
2. Run:
  ```bash
  # --> set $BASELINE_KMEANS50BIG_CHECKPOINT_PATH as the path to CPC-big k-means checkpoint under $ZEROSPEECH2021_BASELINE_PATH/checkpoints

  ./quantize_baseline.sh $LIBRISPEECH_DATASET_PATH flac $ZEROSPEECH_DATASET_PATH wav \
  $SAVE_DIR $BASELINE_KMEANS50BIG_CHECKPOINT_PATH
  ```

This will leave its results in subfolder(s) under $SAVE_DIR


### Centroid-gravity for ABX

1. Complete "Nullspace experiments" section
2. Complete "Performing k-means clustering on nullspace embeddings and producing quantizations" section
3. Activate `zerospeech2021` env
4. For chosen configuration (see examples below), run:
```bash
./centroidGravitation_and_eval_embeddings.sh \
$ZEROSPEECH_PHONETIC_EMBEDDINGS_ROOT \
$MODEL_WITH_COMPUTED_CLUSTERING_CHECKPOINT \
$SUBMISSIONS_TO_EVALUATE_SAVE_PATH \
$ZEROSPEECH_DATASET_PATH \
$LIST_OF_PUSH_DEGREES \
$CLOSEST_CLUSTER_CHOICE_METHOD \
$NORMALIZE_FOR_PUSH_CHOICE
```

This will leave its results in subfolder(s) under $SAVE_DIR

For example to reproduce centroid-gravitation results from the table in ZeroSpeech submission paper (SAVE_DIR is one used for performing k-means clustering on nullspace embeddings as above):
```bash
# nullspace/cosine/cosine
./centroidGravitation_and_eval_embeddings.sh \
$ZEROSPEECH_NULLSPACE_PHONETIC_EMBEDDINGS_ROOT \
$SAVE_DIR/trained_nullspace_cosine_kmeans/kmeans50checkpoint.pt \
$SAVE_DIR/centroid-gravitation-abx-eval/nullspace-cosine-cosine \
$ZEROSPEECH_DATASET_PATH \
"0.2 0.3 0.4 0.5 0.6 0.7" \
cosineclosest \
dontnormalizeforpush

# nullspace/euclidean/cosine
./centroidGravitation_and_eval_embeddings.sh \
$ZEROSPEECH_NULLSPACE_PHONETIC_EMBEDDINGS_ROOT \
$SAVE_DIR/trained_nullspace_euclidean_kmeans/kmeans50checkpoint.pt \
$SAVE_DIR/centroid-gravitation-abx-eval/nullspace-euclid-cosine \
$ZEROSPEECH_DATASET_PATH \
"0.2 0.3 0.4 0.5 0.6 0.7" \
cosineclosest \
dontnormalizeforpush

# no nullspace/euclidean/cosine
./centroidGravitation_and_eval_embeddings.sh \
$SAVE_DIR/reproduce_baseline_ABX_submission/phonetic \
$BASELINE_KMEANS50BIG_CHECKPOINT_PATH \
$SAVE_DIR/centroid-gravitation-abx-eval/nonullspace-euclid-cosine \
$ZEROSPEECH_DATASET_PATH \
"0.2 0.3 0.4 0.5 0.6 0.7" \
cosineclosest \
dontnormalizeforpush
```
To compute for other configurations we tried mentioned and not presented in the paper:
```bash
# nullspace/cosine/euclidean
./centroidGravitation_and_eval_embeddings.sh \
$ZEROSPEECH_NULLSPACE_PHONETIC_EMBEDDINGS_ROOT \
$SAVE_DIR/trained_nullspace_cosine_kmeans/kmeans50checkpoint.pt \
$SAVE_DIR/centroid-gravitation-abx-eval/nullspace-cosine-euclid \
$ZEROSPEECH_DATASET_PATH \
"0.2 0.3 0.4 0.5 0.6 0.7" \
euclideanclosest \
dontnormalizeforpush

# nullspace/euclidean/euclidean
./centroidGravitation_and_eval_embeddings.sh \
$ZEROSPEECH_NULLSPACE_PHONETIC_EMBEDDINGS_ROOT \
$SAVE_DIR/trained_nullspace_euclidean_kmeans/kmeans50checkpoint.pt \
$SAVE_DIR/centroid-gravitation-abx-eval/nullspace-euclid-euclid \
$ZEROSPEECH_DATASET_PATH \
"0.2 0.3 0.4 0.5 0.6 0.7" \
euclideanclosest \
dontnormalizeforpush

# nullspace/cosine/cosine + normpush (normalize before push 
# - approximate moving part of cosine and not euclidean distance)
./centroidGravitation_and_eval_embeddings.sh \
$ZEROSPEECH_NULLSPACE_PHONETIC_EMBEDDINGS_ROOT \
$SAVE_DIR/trained_nullspace_cosine_kmeans/kmeans50checkpoint.pt \
$SAVE_DIR/centroid-gravitation-abx-eval/nullspace-cosine-cosine-normpush \
$ZEROSPEECH_DATASET_PATH \
"0.1 0.2 0.3 0.4 0.5 0.6" \
cosineclosest \
normalizeforpush

# no nullspace/euclidean/euclidean
./centroidGravitation_and_eval_embeddings.sh \
$SAVE_DIR/reproduce_baseline_ABX_submission/phonetic \
$BASELINE_KMEANS50BIG_CHECKPOINT_PATH \
$SAVE_DIR/centroid-gravitation-abx-eval/nonullspace-euclid-euclid \
$ZEROSPEECH_DATASET_PATH \
"0.2 0.3 0.4 0.5 0.6 0.7" \
euclideanclosest \
dontnormalizeforpush
```



### Centroid-gravitation phoneme classification accuracy

In order to reproduce centroid-gravitation phoneme classification accuracy results from paper:

1. Complete "Nullspace experiments" section
2. Complete "Performing k-means clustering on nullspace embeddings and producing quantizations" section
3. Activate `CPC_audio` env
4. For no-nullspace phoneme classification with centroid-gravitation (Euclidean K-Means, Euclidean-closest cluster assignment) run:
```bash
# --> set $BASELINE_NO_CLUSTERING_CHECKPOINT_PATH as the path to CPC-nig checkpoint without clustering data under $ZEROSPEECH2021_BASELINE_PATH/checkpoints
# --> set $BASELINE_KMEANS50BIG_CHECKPOINT_PATH as the path to CPC-big k-means checkpoint under $ZEROSPEECH2021_BASELINE_PATH/checkpoints

./centroidGravitation_nonullspace_phoneme_classification.sh \
$LIBRISPEECH_DATASET_PATH \
$LIBRISPEECH_TRAIN_CLEAN_100_TRAIN_SPLIT_FILE_PATH \
$LIBRISPEECH_TEST_CLEAN_100_TRAIN_SPLIT_FILE_PATH \
$BASELINE_NO_CLUSTERING_CHECKPOINT_PATH \
$BASELINE_KMEANS50BIG_CHECKPOINT_PATH \
$PHONEME_ALIGNMENTS_FILE \
$SAVE_DIR
```

For nullspace phoneme classification with centroid-gravitation (Euclidean K-Means, Euclidean-closest cluster assignment) run:

```bash
# --> set $BASELINE_NO_CLUSTERING_CHECKPOINT_PATH as the path to CPC-nig checkpoint without clustering data under $ZEROSPEECH2021_BASELINE_PATH/checkpoints

./centroidGravitation_nullspace_phoneme_classification.sh \
$LIBRISPEECH_DATASET_PATH \
$LIBRISPEECH_TRAIN_CLEAN_100_TRAIN_SPLIT_FILE_PATH \
$LIBRISPEECH_TEST_CLEAN_100_TRAIN_SPLIT_FILE_PATH \
$BASELINE_NO_CLUSTERING_CHECKPOINT_PATH \
$SAVE_DIR/trained_nullspace_euclidean_kmeans/kmeans50checkpoint.pt
$SAVE_DIR/nullspaces/448/speakers_factorized_448/checkpoint9.pt \
$PHONEME_ALIGNMENTS_FILE \
$SAVE_DIR

```

### Producing cluster distance matrix for sWUGGY

1. Complete "Nullspace experiments" section
2. Complete "Performing k-means clustering on nullspace embeddings and producing quantizations" section
3. Use `dist_matrix_from_clusters.py` script with chosen config (described in the script)
4. Compute chosen power of the matrix if not linear or squared (options provided in the script)


