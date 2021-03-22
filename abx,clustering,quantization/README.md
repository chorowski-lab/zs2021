

This is draft description of how we:
  - produced nullspace-based embeddings
     - (this is a very approximate and undetailed description for now) the code from the repo and branch <https://github.com/chorowski-lab/CPC_audio/tree/jdzikowski/nullspace_loading> and also <https://github.com/chorowski-lab/CPC_audio/tree/jdzikowski/zerospeech> has been used to compute the nullspace
  - produced clustering in the nullspace
    - Code from <https://github.com/chorowski-lab/CPC_audio/tree/jdzikowski/zerospeech> has been used for clustering and quantization. `cpc/criterion/clustering/clustering_script.py` script from there has been used for computing the k-means clustering based on cosine lengths - embeddings and cluster centers were normalized for computing distances, and resulting clusters were normalized (see `cluster_cosine.py` script in this folder). For cluster assignment, `scripts/quantize_audio.py` from the mentioned repo and branch has been used, also with normalizing lengths of embeddings for cosine-distance-based assignment (see `cosine_quant_assign_cosine...` scripts in this folder).
  - performed pushing embeddings to closest centers 
    - `closestpushX.py` scripts from this folder were used to compute "pushed" representations. On top of embeddings, we use the centers of the obtained clusters, and we move each embedding a part of the distance (e.g. half) in the direction of the closest cluster’s center. This aims to include information about the whole dataset coming from clustering without substantial loss of local information, as a kind of denoising. (see also `nullCos...sh`, `nullFull...sh` scripts in this folder that were used to do this)
    
Additionally, cluster centers distance matrix to use for sWUGGY task was made (see `dist_matric_from_clusters.py`).

It is also worth mentioning that we found out the following workflow improved the results:
 - compute representations for the whole LibriSpeech-dev/test dataset
 - remove extra files
 - compute other stuff on top of those
in comparison to computing features for datasets used for ZeroSpeech phonetic metric evaluation. This can perhaps be because audio data in LibriSpeech is consecutive and removing parts of it (some files, as in metric-evaluation dataset) may harm LSTM's context (making high-level features out-of-date (by some files))

Additionally to the files here, we made <https://github.com/chorowski-lab/CPC_audio/tree/jdzikowski/nullspace_loading> and also <https://github.com/chorowski-lab/CPC_audio/tree/jdzikowski/zerospeech>. Those repos use parts or full repos: <https://github.com/facebookresearch/CPC_audio>, <https://github.com/facebookresearch/CPC_audio/tree/zerospeech>, <https://github.com/tuanh208/CPC_audio/tree/zerospeech>,  <https://github.com/bootphon/zerospeech2021_baseline>
