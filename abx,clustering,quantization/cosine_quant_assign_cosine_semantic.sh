

python ../../CPC_audio/scripts/quantize_audio.py \
/pio/gluster/i283340/cosine_quant/nullspace64trainedLStrain-clean-100/trained50clusters.pt \
/pio/data/zerospeech2021/dataset/semantic/dev/librispeech \
/pio/gluster/i283340/cosine_clusters_cosine_assignments/semantic/dev/librispeech/ \
--nobatch --nullspace --norm_vec_len --file_extension wav

python ../../CPC_audio/scripts/quantize_audio.py \
/pio/gluster/i283340/cosine_quant/nullspace64trainedLStrain-clean-100/trained50clusters.pt \
/pio/data/zerospeech2021/dataset/semantic/dev/synthetic \
/pio/gluster/i283340/cosine_clusters_cosine_assignments/semantic/dev/synthetic/ \
--nobatch --nullspace --norm_vec_len --file_extension wav

python ../../CPC_audio/scripts/quantize_audio.py \
/pio/gluster/i283340/cosine_quant/nullspace64trainedLStrain-clean-100/trained50clusters.pt \
/pio/data/zerospeech2021/dataset/semantic/test/librispeech \
/pio/gluster/i283340/cosine_clusters_cosine_assignments/semantic/test/librispeech/ \
--nobatch --nullspace --norm_vec_len --file_extension wav

python ../../CPC_audio/scripts/quantize_audio.py \
/pio/gluster/i283340/cosine_quant/nullspace64trainedLStrain-clean-100/trained50clusters.pt \
/pio/data/zerospeech2021/dataset/semantic/test/synthetic \
/pio/gluster/i283340/cosine_clusters_cosine_assignments/semantic/test/synthetic/ \
--nobatch --nullspace --norm_vec_len --file_extension wav
