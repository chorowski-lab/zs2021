

python ../../CPC_audio/scripts/quantize_audio.py \
/pio/gluster/i283340/cosine_quant/nullspace64trainedLStrain-clean-100/trained50clusters.pt \
/pio/data/zerospeech2021/dataset/lexical/dev \
/pio/gluster/i283340/cosine_clusters_cosine_assignments/lexical/dev/ \
--batch_size 128 --nullspace --norm_vec_len --file_extension wav

python ../../CPC_audio/scripts/quantize_audio.py \
/pio/gluster/i283340/cosine_quant/nullspace64trainedLStrain-clean-100/trained50clusters.pt \
/pio/data/zerospeech2021/dataset/lexical/test \
/pio/gluster/i283340/cosine_clusters_cosine_assignments/lexical/test/ \
--batch_size 128 --nullspace --norm_vec_len --file_extension wav