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
    zerospeech2021-evaluate --force-cpu --no-lexical --no-syntactic --no-semantic -j 20 -o $outputDir $datasetDir $submDir >> $outFile
done

