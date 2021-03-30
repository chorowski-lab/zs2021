


python closestpushClean.py \
/pio/data/zerospeech2021/checkpoints/CPC-big-kmeans50/clustering_kmeans50/clustering_CPC_big_kmeans50.pt \
/pio/gluster/i283340/checkpushsubmsClean/comparetofcm_Cos \
dev-clean{}../features_lvl2_all-ls_devclean:dev-other{}../features_lvl2_all-ls_devother \
/pio/gluster/i283340/subms_null/subm_nullspaceCosClustCosAssign_deg0-6/meta.yaml \
32 \
16 \
no \
0.2:0.3:0.4:0.5:0.6:0.7 \
True \
False

./evalmore.sh /pio/gluster/i283340/checkpushsubmsClean/comparetofcm_Cos_ /pio/gluster/i283340/checkpushsubmsClean/eval_comparetofcm_Cos_ "deg0-2 deg0-3 deg0-4 deg0-5 deg0-6 deg0-7" /pio/scratch/1/i283340/MGR/zs/quantization/ls-ds-dev/ /pio/gluster/i283340/checkpushsubmsClean/out_comparetofcm_Cos.txt

