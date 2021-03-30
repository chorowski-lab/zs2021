
# cos clustering, cos assignment, nullspace

python closestpushClean.py \
/pio/gluster/i283340/cosine_quant/nullspace64trainedLStrain-clean-100/trained50clusters.pt \
/pio/gluster/i283340/subms_nullClean/subm_nullspaceCosClustCosAssign_normPush \
dev-clean{}../null-features/phonetic/dev-clean:dev-other{}../null-features/phonetic/dev-other \
/pio/gluster/i283340/subms_null/subm_nullspaceCosClustCosAssign_deg0-6/meta.yaml \
32 \
20 \
no \
0.1:0.2:0.3:0.4:0.5:0.6 \
True \
True

./evalmore.sh /pio/gluster/i283340/subms_nullClean/subm_nullspaceCosClustCosAssign_normPush_ /pio/gluster/i283340/subms_nullClean/eval_nullspaceCosClustCosAssign_normPush_ "deg0-1 deg0-2 deg0-3 deg0-4 deg0-5 deg0-6" /pio/scratch/1/i283340/MGR/zs/quantization/ls-ds-dev/ /pio/gluster/i283340/subms_nullClean/out_nullspaceCosClustCosAssign_normPush.txt


