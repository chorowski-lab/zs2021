
# cos clustering, cos assignment, nullspace

python closestpush.py \
/pio/gluster/i283340/cosine_quant/nullspace64trainedLStrain-clean-100/trained50clusters.pt \
/pio/gluster/i283340/subms_null/subm_nullspaceCosClustCosAssign \
dev-clean{}../null-features/phonetic/dev-clean:dev-other{}../null-features/phonetic/dev-other \
../subm_fcm3w_m1-5_deg0-75/meta.yaml \
32 \
20 \
no \
0.6:0.75 \
True

./evalmore.sh /pio/gluster/i283340/subms_null/subm_nullspaceCosClustCosAssign_ /pio/gluster/i283340/subms_null/eval_3nullspaceCosClustCosAssign_ "deg0-6 deg0-75" /pio/scratch/1/i283340/MGR/zs/quantization/ls-ds-dev/ /pio/gluster/i283340/subms_null/out_3nullspaceCosClustCosAssign.txt
#./evalmore.sh /pio/gluster/i283340/checkpushsubms/comparetofcm_ /pio/gluster/i283340/checkpushsubms/evalcomparetofcm_ "deg0-5" /pio/scratch/1/i283340/MGR/zs/quantization/ls-ds-dev/ /pio/gluster/i283340/checkpushsubms/compare0-5.txt