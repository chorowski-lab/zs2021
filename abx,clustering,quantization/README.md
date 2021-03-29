
*NOTE: paths in scripts in this directory may be inaccurate as those srcipts were copied from another directory structure, but it should be straightforward to figure out which other scripts etc. they call*

In the `CPC_audio` folder there is a snapshot of our <https://github.com/chorowski-lab/CPC_audio>, which also uses parts or full repos: <https://github.com/facebookresearch/CPC_audio>, <https://github.com/facebookresearch/CPC_audio/tree/zerospeech>, <https://github.com/tuanh208/CPC_audio/tree/zerospeech>,  <https://github.com/bootphon/zerospeech2021_baseline>. For more details see `CPC_audio` folder and `CPC_audio/README.md`.

The code in this folder has been used for:
  - producing nullspace-based embeddings
     - the code from the `CPC_audio` folder has been used for nullspace technique; please see `CPC_audio/README.md` for the description (in the "Nullspaces" subsection). Additionally to check phoneme separability for the nullspaces, `CPC_audio/finetune_nullspace.sh` was used.
  - producing clustering in the nullspace
    - Code from `CPC_audio` folder has been used for clustering and quantization. `CPC_audio/cpc/criterion/clustering/clustering_script.py` script from there has been used for computing the k-means clustering based on cosine lengths - embeddings and cluster centers were normalized for computing distances, and resulting clusters were normalized too (see `cluster_cosine.sh` script in this folder). For cluster assignment, `CPC_audio/scripts/quantize_audio.py` has been used, also with normalizing lengths of embeddings for cosine-distance-based assignment (see `cosine_quant_assign_cosine...` scripts in this folder). For basline quantizations we also used `CPC_audio/scripts/quantize_audio.py` (taken from zerospeech2021_baseline repo).
  - performing pushing embeddings to closest centers 
    - `closestpushCelan.py` script from this folder were used to compute "pushed" representations. On top of embeddings, we use the centers of the obtained clusters, and we move each embedding a part of the distance (e.g. half) in the direction of the closest cluster’s center. This aims to include information about the whole dataset coming from clustering without substantial loss of local information, as a kind of denoising. (`nullCosSpCosClean.sh` and similar scripts in this folder that were used to do this and evaluate ABX; we were pushing a chosen part of euclidean distance for various euclidean/cosine k-means clustering and euclidean/cosine closest cluster choice combinations; additionally `noNullPushClean.sh` operates on original embeddings without the nullspace technique and `nullCosSpCosCleanNormPush.sh` normalizes embeddings before pushing which better approximates pushing a part of cosine and not euclidean distance, but results were similar after tuning). Additionally, `CPC_audio/centerpush...phoneme_classification.sh` scripts were used for phoneme separability experiment with center-pushing.
    
Additionally, cluster centers distance matrix to use for sWUGGY task was made (see `dist_matrix_from_clusters.py`).

It is also worth mentioning that we found out the following workflow improved the results:
 - compute representations for the whole LibriSpeech-dev/test dataset
 - remove extra files
 - compute other things on top of those
in comparison to computing features for datasets used for ZeroSpeech phonetic metric evaluation. This can perhaps be because audio data in LibriSpeech is (at least sometimes) consecutive and removing parts of it (some files, as in ZeroSpeech ABX-evaluation dataset) may harm autoregressive context (as `zerospeech2021_baseline/scripts/build_CPC_features.py` script we used for building features keeps autoregressive context between files as default, so removing some consecutive-audio files was perhaps making high-level features out-of-date (by some files))
