

python ../../CPC_audio/scripts/quantize_audio.py \
/pio/gluster/i283340/cosine_quant/nullspace64trainedLStrain-clean-100/trained50clusters.pt  \
/pio/data/zerospeech2021/LibriSpeech/dev-clean \
/pio/gluster/i283340/cosine_clusters_cosine_assignments/LibriSpeech/dev-clean/ \
--nullspace --norm_vec_len --file_extension flac

python ../../CPC_audio/scripts/quantize_audio.py \
/pio/gluster/i283340/cosine_quant/nullspace64trainedLStrain-clean-100/trained50clusters.pt  \
/pio/data/zerospeech2021/LibriSpeech/dev-other \
/pio/gluster/i283340/cosine_clusters_cosine_assignments/LibriSpeech/dev-other/ \
--nullspace --norm_vec_len --file_extension flac

python ../../CPC_audio/scripts/quantize_audio.py \
/pio/gluster/i283340/cosine_quant/nullspace64trainedLStrain-clean-100/trained50clusters.pt  \
/pio/data/zerospeech2021/LibriSpeech/test-clean \
/pio/gluster/i283340/cosine_clusters_cosine_assignments/LibriSpeech/test-clean/ \
--nullspace --norm_vec_len --file_extension flac

python ../../CPC_audio/scripts/quantize_audio.py \
/pio/gluster/i283340/cosine_quant/nullspace64trainedLStrain-clean-100/trained50clusters.pt  \
/pio/data/zerospeech2021/LibriSpeech/test-other \
/pio/gluster/i283340/cosine_clusters_cosine_assignments/LibriSpeech/test-other/ \
--nullspace --norm_vec_len --file_extension flac

python ../../CPC_audio/scripts/quantize_audio.py \
/pio/gluster/i283340/cosine_quant/nullspace64trainedLStrain-clean-100/trained50clusters.pt  \
/pio/data/zerospeech2021/LibriSpeech/train-clean-100 \
/pio/gluster/i283340/cosine_clusters_cosine_assignments/LibriSpeech/train-clean-100/ \
--nullspace --norm_vec_len --file_extension flac --batch_size 32

python ../../CPC_audio/scripts/quantize_audio.py \
/pio/gluster/i283340/cosine_quant/nullspace64trainedLStrain-clean-100/trained50clusters.pt  \
/pio/data/zerospeech2021/LibriSpeech/train-clean-360 \
/pio/gluster/i283340/cosine_clusters_cosine_assignments/LibriSpeech/train-clean-360/ \
--nullspace --norm_vec_len --file_extension flac --batch_size 32

python ../../CPC_audio/scripts/quantize_audio.py \
/pio/gluster/i283340/cosine_quant/nullspace64trainedLStrain-clean-100/trained50clusters.pt  \
/pio/data/zerospeech2021/LibriSpeech/train-full-960 \
/pio/gluster/i283340/cosine_clusters_cosine_assignments/LibriSpeech/train-full-960/ \
--nullspace --norm_vec_len --file_extension flac --batch_size 32