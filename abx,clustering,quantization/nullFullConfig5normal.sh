

python closestpush3.py \
/pio/gluster/i283340/cosine_quant/nullspace64trainedLStrain-clean-100/trained50clusters.pt \
/pio/gluster/i283340/subms_null/subm_FULL_3nullspaceCosClustCosAssign \
dev-clean{}/pio/gluster/i283340/null-features-all/phonetic/dev-clean:dev-other{}/pio/gluster/i283340/null-features-all/phonetic/dev-other:test-clean{}/pio/gluster/i283340/null-features-all/phonetic/test-clean:test-other{}/pio/gluster/i283340/null-features-all/phonetic/test-other \
../subm_fcm3w_m1-5_deg0-75/meta.yaml \
32 \
20 \
no \
0.5 \
True \
False

./evalmore.sh /pio/gluster/i283340/subms_null/subm_FULL_3nullspaceCosClustCosAssign_ /pio/gluster/i283340/subms_null/eval_FULL_3nullspaceCosClustCosAssign_ "deg0-5" /pio/data/zerospeech2021/dataset/ /pio/gluster/i283340/subms_null/out_FULL_3nullspaceCosClustCosAssign.txt
#./evalmore.sh /pio/gluster/i283340/checkpushsubms/comparetofcm_ /pio/gluster/i283340/checkpushsubms/evalcomparetofcm_ "deg0-5" /pio/scratch/1/i283340/MGR/zs/quantization/ls-ds-dev/ /pio/gluster/i283340/checkpushsubms/compare0-5.txt






