echo "List: $3"

baseSubmPath=$1
baseOutputPath=$2
datasetDir=$4
outFile=$5
for item in $3 ; do  # passing list args:   "a b c"
    submDir="${baseSubmPath}${item}"
    outputDir="${baseOutputPath}${item}"
    echo $submDir
    echo $outputDir
    mkdir $outputDir
    echo $outputDir >> $outFile
    zerospeech2021-evaluate --force-cpu --no-lexical --no-syntactic --no-semantic -j 20 -o $outputDir $datasetDir $submDir >> $outFile
done

