
python ../../CPC_audio/cpc/criterion/clustering/clustering_script.py \
--pathDB /pio/data/zerospeech2021/LibriSpeech/train-clean-100/ \
--recursionLevel 1 --nClusters 50 --MAX_ITER 150 --level_gru 2 --save --load \
--batchSizeGPU 500 --max_size_loaded 40000000 --n_process_loader 2 \
--nullspace --norm_vec_len \
/pio/scratch/1/i273233/linear_separability/cpc/gru_level2/cpc_official_phonemes_nullspace_64/checkpoint_9.pt \
/pio/gluster/i283340/cosine_quant/nullspace64trainedLStrain-clean-100/trained50clusters.pt
