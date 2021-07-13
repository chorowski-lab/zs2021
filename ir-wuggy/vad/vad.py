import argparse
import csv
import os
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
import webrtcvad
import yaml
from more_itertools import split_at
from tqdm import tqdm


class VADDataset(torch.utils.data.Dataset):
    def __init__(self, audio_data_folder: str, quant_folder: str, frame_duration: int = 10):
        self.audio_data_folder = audio_data_folder
        self.quant_data = pd.read_csv(os.path.join(quant_folder, 'quantized_outputs.txt'), delimiter='\t')
        assert frame_duration in [10, 20, 30], 'webrtc requires frames to be either 10, 20, or 30ms'
        self.frame_duration = frame_duration

    def __getitem__(self,idx: int) -> Tuple[str, np.ndarray]:
        encoded_file_path, quant_audio = self.quant_data.iloc[idx]
        quant_audio = np.array([int(phoneme) for phoneme in quant_audio.split(',')])
        # splitting by '-' should give #1 folder id, #2 folder id, track id
        encoded_path_split = encoded_file_path.split('-')[:2]
        full_relative_path = os.path.join(self.audio_data_folder, *encoded_path_split, encoded_file_path + '.flac')
        # read audio and quantised counterpart, pass further
        audio_content, sample_rate = torchaudio.load(full_relative_path)
        audio_content = audio_content.flatten()
        sample_size = int(sample_rate * self.frame_duration / 1000)
        # we assume that there will be no phoneme in frame shrter than 10ms
        cut_size =  int((np.floor(len(audio_content) / sample_size) * sample_size))
        audio_content = audio_content[:cut_size]
        return encoded_file_path, quant_audio,  audio_content

    def __len__(self):
        return len(self.quant_data)


def parseArgs():
    parser = argparse.ArgumentParser(description='Incorporate VAD signal into data')
    parser.add_argument('--config', type=str,
                        help='Location of the .yaml config file')
    return parser.parse_args()

def parseConfig(args):
    with open(args.config) as config_file:
        return yaml.full_load(config_file)

def predict_vad(vad_obj: webrtcvad.Vad, audio) -> List[bool]:
    pred_signal = []
    offset = 0
    n = 160
    while offset + n <= len(audio):
        pred_signal.append(vad_obj.is_speech(audio[offset:offset + n].tobytes(), sample_rate=16000))
        offset += n
    return pred_signal


def find_splits(vad_signal: List[bool], quant_audio: np.ndarray):
    splits_en = [speech for speech in list(split_at(enumerate(quant_audio), lambda x: not vad_signal[x[0]])) if speech]
    splits = []
    for split in splits_en:
        splits.append([elem[1] for elem in split])
    return splits



def write_fn_separate(writer: csv.writer , filename: str, splits: List[int]):
    """After separation, splits of each example becomes new examples (are written in new lines, no additional phoneme is introduced)"""

    for idx, split in enumerate(splits):
        indexed_filename = '-'.join([filename, str(idx)])
        audio_str = ','.join(map(str, split))
        writer.writerow([indexed_filename, audio_str])

def write_fn_replace(writer: csv.writer , filename: str, splits: List[int]):
    """Replace silence with additional '*' phoneme"""

    reinputed_sequence = []
    reinputed_sequence.extend(splits[0])
    for split in splits[1:]:
        reinputed_sequence.append('*')
        reinputed_sequence.extend(split)

    audio_str = ','.join(reinputed_sequence)
    writer.write_row([filename, audio_str])


def main(write_fn: Callable[[csv.writer, str, List[int]], None], config):
    test_dataset = VADDataset(audio_data_folder=config['audioPath'],
                              quant_folder=config['quantPath'])
    test_dataloader = torch.utils.data.DataLoader(test_dataset)

    vad_predictor = webrtcvad.Vad()

    Path(config['outPath']).mkdir(parents=True, exist_ok=True)
    output_file_path = os.path.join(config['outPath'], 'quantized.txt')

    with open(output_file_path, 'w') as out_fp:
        writer = csv.writer(out_fp, delimiter='\t')
        for encoded_file_path, quant_audio,  audio_content in tqdm(test_dataloader):
            quant_audio = quant_audio.flatten().tolist()
            audio_content = audio_content.flatten().numpy()
            vad_signal = predict_vad(vad_obj=vad_predictor, audio=audio_content)
            splits = find_splits(vad_signal, quant_audio)
            write_fn(writer=writer, filename=encoded_file_path[0], splits=splits)


if __name__ == "__main__":
    args = parseArgs()
    config = parseConfig(args)
    print(config)
    if config['method'] =='split':
        write_fn = write_fn_separate
    elif config['method'] == 'replace':
        write_fn = write_fn_replace
    else:
        ValueError('Unrecognised write method')
    main(write_fn = write_fn, config=config)
