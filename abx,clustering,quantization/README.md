
*NOTE: paths in scripts in this directory may be inaccurate as those srcipts were copied from another directory structure, but it should be straightforward to figure out which other scripts etc. they call*

In the `CPC_audio` folder there is a snapshot of our <https://github.com/chorowski-lab/CPC_audio>, which also uses parts or full repos: <https://github.com/facebookresearch/CPC_audio>, <https://github.com/facebookresearch/CPC_audio/tree/zerospeech>, <https://github.com/tuanh208/CPC_audio/tree/zerospeech>,  <https://github.com/bootphon/zerospeech2021_baseline>. For more details see `CPC_audio` folder and `CPC_audio/README.md`.

The code in this folder has been used for:
  - producing nullspace-based embeddings
  - performing Euclidean and cosine k-means clustering in the nullspace
  - producing embeddings' quantizations for tasks other than ABX
  - using 'centroid-gravitation'
    
Additionally, cluster centers distance matrix to use for sWUGGY task was made (see `dist_matrix_from_clusters.py`).


## How to reproduce the results:

Names written as variables in capital letters are to be changed to user-specific when executing this


### Reproducing baseline ABX with workflow described in the submission paper:

We were only able to reproduce baseline results with following workflow:
 - compute representations for the whole LibriSpeech-dev/test dataset
 - remove extra files
 - compute other things on top of those
which achieved better results in comparison to computing features for datasets used for ZeroSpeech phonetic metric evaluation. This can perhaps be because audio data in LibriSpeech tends to be consecutive and removing parts of it (some files, as in ZeroSpeech ABX-evaluation dataset) may harm autoregressive context (as `zerospeech2021_baseline/scripts/build_CPC_features.py` script we used for building features keeps autoregressive context between files as default, so removing some consecutive-audio files was perhaps making high-level features out-of-date (by some files))

1. Install `zeroseppech2021_baseline` with the environment and download checkpoints as described in its readme
2. Run:
```
./reproduce_baseline_ABX.sh \
$ZEROSPEECH2021_BASELINE_PATH $LIBRISPEECH_DATASET_PATH $ZEROSPEECH_DATASET_PATH \
$SAVE_DIR
```

This will leave produced embeddings under `SAVE_DIR/reproduce_baseline_ABX_submission/phonetic`.


### Nullspace experiments

1. Install CPC_audio with the environment (version under CPC_audio folder here) and soundfile with pip instead of conda one and activate it
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
        -f FROM_STEP (From which step do you want to start. Order: speakers_factorized [default] -> phonemes_nullspace -> speakers_nullspace)
        -p PHONEME_ALIGNMENTS_FILE (Path to the file containing phonemes for the entire dataset. You don't need it if you start from speakers_nullspace)
```
In order to reproduce our experiment from the paper, run the following:

```bash
for i in 256 320 416 448 464
do
    ./finetune_nullspace.sh -d $LIBRISPEECH_DATASET_PATH/train-clean-100 -t $LIBRISPEECH_TRAIN_CLEAN_100_TRAIN_SPLIT_FILE_PATH -v $LIBRISPEECH_TRAIN_CLEAN_100_TEST_SPLIT_FILE_PATH -c $BASELINE_NO_CLUSTERING_CHECKPOINT_PATH -o $SAVE_DIR/$i -n $(expr 512 - $i) -p $PHONEME_ALIGNMENTS_FILE
done
```

### Evaluating ABX for nullspace

1. Install `zerospeech2021` environment
2. Install CPC_audio with the environment (version under CPC_audio folder here) and soundfile with pip instead of conda one and activate it
3. Create the LibriSpeech dev/test dataset. Once you have done this you do not have to do it anymore:
```bash
for directory in dev-clean dev-other test-clean test-other
do
  mkdir -p $LIBRISPEECH_FLATTENED_DATASET_PATH/phonetic/$directory
  cp $LIBRISPEECH_DATASET_PATH/$directory/*/*/*.wav $LIBRISPEECH_FLATTENED_DATASET_PATH/phonetic/$directory
done

for directory in dev-clean dev-other
do
  cp $ZEROSPEECH_DATASET_PATH/phonetic/$directory/$directory.item $LIBRISPEECH_FLATTENED_DATASET_PATH/phonetic/$directory
done
```
4. Run `scripts/eval_abx.sh` from `CPC_audio` directory:

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
```
In order to reproduce ABX error rates for a CPC + nullspace, run the following (Note that LIBRISPEECH_FLATTENED_DATASET_PATH refers to the dateset created earlier and not to the path where LibriSpeech is located):

```bash
scripts/eval_abx.sh -d $ZEROSPEECH_DATASET_PATH -r $ZEROSPEECH_DATASET_PATH -c $CHECKPOINT_PATH -o $SAVE_DIR/original -n -a $CONDA_PATH -e $CPC_ENVIRONMENT -z $ZEROSPEECH_EVAL_ENVIRONMENT

scripts/eval_abx.sh -d $LIBRISPEECH_FLATTENED_DATASET_PATH -r $ZEROSPEECH_DATASET_PATH -c $CHECKPOINT_PATH -o $SAVE_DIR/librispeech -n -a $CONDA_PATH -e $CPC_ENVIRONMENT -z $ZEROSPEECH_EVAL_ENVIRONMENT

```


### Performing k-means clustering on nullspace embeddings and producing quantizations:

1. Install CPC_audio with the environment (version under `CPC_audio` folder here) and soundfile with pip instead of conda one and activate it
2. For cosine k-means clustering and cosine-closest assignment quantizations run:
  ```
  ./cluster_nullspace_cosine_and_quantize.sh \
  $LIBRISPEECH_DATASET_PATH flac $ZEROSPEECH_DATASET_PATH wav \
  $SAVE_DIR $NULLSPACE_MODEL_NO_CLUSTERING_CHECKPOINT_PATH
  ```
  For euclidean k-means clustering and euclidean-closest assignment quantizations run:
  ```
  ./cluster_nullspace_euclidean_and_quantize.sh \
  $LIBRISPEECH_DATASET_PATH flac $ZEROSPEECH_DATASET_PATH wav \
  $SAVE_DIR $NULLSPACE_MODEL_NO_CLUSTERING_CHECKPOINT_PATH
  ```



### Producing baseline quantizations

1. Install CPC_audio with the environment (version under `CPC_audio` folder here) and soundfile with pip instead of conda one and activate it
2. Run:
  ```
  ./quantize_baseline.sh $LIBRISPEECH_DATASET_PATH flac $ZEROSPEECH_DATASET_PATH wav \
  $SAVE_DIR $BASELINE_KMEANS50BIG_CHECKPOINT_PATH
  ```



### Centroid-gravity for ABX

1. Install and activate `zerospeech2021` environment
2. For chosen configuration (see examples below), run:
```
./centroidGravitation_and_eval_embeddings.sh \
$ZEROSPEECH_PHONETIC_EMBEDDINGS_ROOT \
$MODEL_WITH_COMPUTED_CLUSTERING_CHECKPOINT \
$SUBMISSIONS_TO_EVALUATE_SAVE_PATH \
$ZEROSPEECH_DATASET_PATH \
$LIST_OF_PUSH_DEGREES \
$CLOSEST_CLUSTER_CHOICE_METHOD \
$NORMALIZE_FOR_PUSH_CHOICE
```
For example to reproduce centroid-gravitation for reproducing the results from the table in submission paper (SAVE_DIR is one used for performing k-means clustering on nullspace embeddings as above):
```
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
```
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

1. Download LibriSpeech dataset
2. Create/download LibriSpeech train-clean-100 split into train and test sets saved in .txt files where each line contains name of file belonging in this split part (e.g. 7780-274562-0073 etc.)
3. Download phoneme alignments in correct format mentioned in `CPC_audio` readme
4. Make nullspace models as described in this readme
5. Make nullspace and no-nullspace k-means clusterings as described in this readme
6. For no-nullspace phoneme classification with centroid-gravitation (Euclidean K-Means, Euclidean-closest cluster assignment) run:
```
./centroidGravitation_nonullspace_phoneme_classification.sh \
$LIBRISPEECH_DATASET_PATH \
$LIBRISPEECH_TRAIN_CLEAN_100_TRAIN_SPLIT_FILE_PATH \
$LIBRISPEECH_TEST_CLEAN_100_TRAIN_SPLIT_FILE_PATH \
$BASELINE_NO_CLUSTERING_CHECKPOINT_PATH \
BASELINE_KMEANS50BIG_CHECKPOINT_PATH \
$PHONEME_ALIGNMENTS_FILE \
$SAVE_DIR
```

For nullspace phoneme classification with centroid-gravitation (Euclidean K-Means, Euclidean-closest cluster assignment) run:

```
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