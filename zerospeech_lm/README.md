# Language models for ZeroSpeech2021 submission by Univeristy of Wrocław
The repository contains scripts for training and evaluating language models on sWUGGY and sBLIMP tasks (see [ZeroSpeech2021](https://zerospeech.com/2021/news.html) website for details).

## Requirements
Stored in `environment.yml`.

## Training
Follow intructions from https://github.com/bootphon/zerospeech2021_baseline to extract features or quantizations of LibriSpeech training split.

Our LM scripts use [Hydra](https://github.com/facebookresearch/hydra) for passing arguments directly from `.yaml` files instead of through command line. 
To train a new model, simply set paths in the `configs/config.yaml` file, and/or modify model's architecture. Then run
```
python scripts/train.py
```

## Evaluation
Run
```
python scripts/evaluation.py
```
to evaluate the model on sWUGGY and sBLIMP tasks. The above script creates valid submission folder with task-specific subfolders. You can then evaluate scores using scripts provided in https://github.com/bootphon/zerospeech2021.
