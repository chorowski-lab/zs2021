# DTW for the Lexical task of Zerospeech 2021

## Computing DTW

### Prerequisities

1. Quantized data set, probably LibriSpeech. Example: `/pio/data/zerospeech2021/quantized/LibriSpeech/train-full-960` (their baseline)

2. Quantized dev/test set of the lexical task. Example: `/pio/data/zerospeech2021/quantized/lexical/dev-big-no-oov`(their baseline, without the OOV part)

3. (optional) Distance matrix between pseudophonemes. Examples at `dm` folder.

### How to run

1. Create a configuration file (examples are in the `configurations` folder):
```yaml
trainPath: <path-to-the-data-set>
testPath: <path-to-the-test-set>
outPath: <output-path>/dev                      # here will be files 'dev-{i}' created
method:
  name: 'dtw'
  distMatrix: <path-to-the-distance-matrix>     # optional, but works better with it
  extended: <n>                                 # if set, it will not round&gather results, but output <n> best matches with corresponding filenames
transform:                                      # optional, to run transformation on train/test sets
  name: 'cleanup' / 'squash'
  <transform-specific-params>                   
saveEvery: 1000                                 # or any other number dividing 40000
```

1. Run `python dtw.py <location-of-the-config-file>`. It computes DTW and writes results to the specified 

2. Run `python concat.py <output-path> [--test] [--method=min|sum|firstk] [--k=<k>] [--n=<n>] [--norm=<norm>]`. We need to create a single `dev.txt`/`test.txt` file for submission, and this script does exactly that. Available parameters:
  - `--test` - if specified, the output file will be named `test.txt` instead of the default `dev.txt`.
  - `--n` - if specified, it will concat only the `dev-1`, ... `dev-n` files. Otherwise it tries to concat all `dev-i` files.
  - `--method` - when applying to non-extended variant (there was no 'extended' in config), it can be either 'min', 'sum' or 'firstk'. When applying to extended variant, only 'min' (the default) is available.
  - `--k` - a parameter for the 'firstk' method. It will take a negative of a sum of the first $k$ DTW scores.
  - `--norm` - a parameter for the 'min' method in the extended variant. If set to $q$, then it will normalize the min DTW score by the $q$-th min DTW score.
