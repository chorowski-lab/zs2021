echo "List: $3"
#x=(a b)
#for item in "${x[@]}" ; do
baseSubmPath=$1
baseOutputPath=$2
datasetDir=$4
outFile=$5
for item in $3 ; do  # passing list args:   "arg1_subarg1 arg1_subarg2"
    submDir="${baseSubmPath}${item}"
    outputDir="${baseOutputPath}${item}"
    echo $submDir
    echo $outputDir
    mkdir $outputDir
    echo $outputDir >> $outFile
    zerospeech2021-evaluate --force-cpu --no-lexical --no-syntactic --no-semantic -j 20 -o $outputDir $datasetDir $submDir >> $outFile   # --force-cpu
done


# ./evalmore.sh ../subm_fcm3w_ ../evalres_fcm3w_ "m1-15_deg0-25 m1-15_deg0-5 m1-15_deg0-75 m1-25_deg0-25 m1-25_deg0-5 m1-25_deg0-75 m1-5_deg0-25 m1-5_deg0-5 m1-5_deg0-75 m1-75_deg0-25 m1-75_deg0-5 m1-75_deg0-75 m2-0_deg0-25 m2-0_deg0-5 m2-0_deg0-75 m2-25_deg0-25 m2-25_deg0-5 m2-25_deg0-75 m2-5_deg0-25 m2-5_deg0-5 m2-5_deg0-75" /pio/data/zerospeech2021/dataset_subset/ ../outeval_fcw3_001.txt
# ./evalmore.sh ../subm_fcm3w_ ../evalres_fcm3w_ "m1-1_deg0-25 m1-1_deg0-5 m1-1_deg0-75" /pio/data/zerospeech2021/dataset_subset/ ../outeval_fcw3_002.txt
# ./evalmore.sh ../subm_fcm3baselinefull_ ../evalres_fcm3baselinefull_ "m0-0_deg0-0" /pio/data/zerospeech2021/dataset/ ../outeval_fcw3baselinefull_001.txt
# ./evalmore.sh ../subm_fcm3wfull_ ../evalres_fcm3wfull_ "m1-1_deg0-5" /pio/data/zerospeech2021/dataset/ ../outeval_fcw3fcmfull_001.txt
# ./evalmore.sh ../subm_fcm3wfull_ ../evalres_fcm3wfull_ "m1-15_deg0-5" /pio/data/zerospeech2021/dataset/ ../outeval_fcw3fcmfull_002.txt

# ./evalmore.sh ../subm_fcm3w_ ../evalres_fcm3w_cos_ "m1-15_deg0-25 m1-15_deg0-5 m1-15_deg0-75 m1-25_deg0-25 m1-25_deg0-5 m1-25_deg0-75 m1-5_deg0-25 m1-5_deg0-5 m1-5_deg0-75 m1-75_deg0-25 m1-75_deg0-5 m1-75_deg0-75 m2-0_deg0-25 m2-0_deg0-5 m2-0_deg0-75 m2-25_deg0-25 m2-25_deg0-5 m2-25_deg0-75 m2-5_deg0-25 m2-5_deg0-5 m2-5_deg0-75" /pio/data/zerospeech2021/dataset_subset/ ../outeval_fcw3_001cosine.txt
# ./evalmore.sh ../subm_fcm3w_baseline_cos_ ../evalres_fcm3w_baseline_cos_ "m0-0_deg0-0" /pio/data/zerospeech2021/dataset_subset/ ../outeval_fcw3w_baseline_cosine_001.txt