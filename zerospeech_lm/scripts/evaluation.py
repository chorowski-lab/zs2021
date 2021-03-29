import os
import hydra
from omegaconf import DictConfig, OmegaConf
import yaml
import pickle
from tqdm import tqdm

import numpy as np
import torch

from zerospeech_lm.models.utils import reset_hidden, load_checkpoint
from zerospeech_lm.models.lm import LSTM_model, QRNN_model
from zerospeech_lm.models.clustering import load_clustering


def eval_model(model, clustering, epoch, args, dev_only=True):
    create_submission_folder(epoch, args)

    splits = ['dev']
    if not dev_only:
        splits.append('test')

    for split in splits:
        model.eval()
        if clustering is None:
            quantized_eval(model, epoch, split, args)
        else:
            feature_eval(model, clustering, epoch, split, args)

def feature_eval(model, clustering, epoch, split, args):
    split_dir = os.path.join(args.eval_data_dir, 'lexical', split)
    hidden_state = reset_hidden(args, eval=True)

    with torch.no_grad(), \
        open(os.path.join(args.save_dir, 'outputs', str(epoch), 'lexical', split + '.txt'), 'w') as out_file:
        for file_name in tqdm(os.listdir(split_dir), desc='Evaluating model on {} split'.format(split), ascii=True):
            if file_name[-4:] != '.npy':
                continue

            seq = np.load(os.path.join(split_dir, file_name)).astype(float)

            clustered_seq = clustering.predict(seq).reshape(1, -1)
            clustered_seq = torch.from_numpy(clustered_seq).to(torch.int64)
            x = clustered_seq[:, :-1].to(args.device)
            y = clustered_seq[:, 1:].to(args.device)
            
            score = 0.
            probs, _ = model(x, hidden_state)
            print(probs.shape)
            for i in range(y.size(1)):
                score += probs[0,i,y[0,i]].item()

            out_file.write('{} {}\n'.format(file_name[:-4], score))

def quantized_eval(model, epoch, split, args):
    hidden_state = reset_hidden(args, eval=True)
    tasks = ['syntactic', 'lexical']
    for task in tasks:
        split_dir = os.path.join(args.eval_data_dir, task, split)

        with torch.no_grad(), \
            open(os.path.join(args.save_dir, 'outputs', str(epoch), task, split + '.txt'), 'w') as out_file, \
            open(os.path.join(split_dir, 'quantized_outputs.txt')) as in_file:

            quants = in_file.readlines()

            for line in tqdm(quants, desc='Evaluating model on {} task {} split'.format(task, split), ascii=True):
                file_name, seq = line.split()
                seq = np.array(seq.split(','), dtype=np.int32)
                seq = torch.tensor(seq).to(torch.int64)
                if args.inverse_seqs:
                    seq = seq.flip(0) # flip sequences for backward training
                x = seq[:-1].reshape(1,-1).to(args.device)
                y = seq[1:].reshape(1,-1).to(args.device)
                
                score = 0.
                probs, _ = model(x, hidden_state)
                for i in range(y.size(1)):
                    score += probs[0,i,y[0,i]].item()

                out_file.write('{} {}\n'.format(file_name, score))

def calculate_entropy(model, data, epoch, args):
    path = os.path.join(args.save_dir, 'entropy', str(epoch))
    if not os.path.exists(path):
        os.makedirs(path)

    hidden_state = reset_hidden(args, eval=True)
    dd = {}
    with torch.no_grad(), \
        open(os.path.join(args.save_dir, 'entropy', str(epoch), 'entropy'), 'wb') as out_file:
        for seq, file_name in tqdm(data, desc='Calculating entropy', ascii=True):
            seq = torch.tensor(seq).to(torch.int64)
            x = seq[:-1].reshape(1,-1).to(args.device)
            probs, _ = model(x, hidden_state)
            entropy = (probs[0] * torch.exp(probs[0])).sum(1).cpu().numpy()
            dd[file_name] = entropy
        pickle.dump(dd, out_file)

def create_submission_folder(epoch, args):
    # main folder
    path = os.path.join(args.save_dir, 'outputs', str(epoch))
    if not os.path.exists(path):
        os.makedirs(path)
    
        # task-specific folders
        for sub_name in ['lexical', 'phonetic', 'semantic', 'syntactic']:
            sub_path = os.path.join(path, sub_name)
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)
    
        # meta.yaml file
        meta_dict = {
            'author': 'author',
            'affiliation': 'affiliation',
            'description': 'description',
            'open_source': False,
            'train_set': 'LibriSpeech',
            'gpu_budget': 1,
            'parameters': {
                'phonetic': {
                    'metric': 'euclidean',
                    'frame_shift': 1
                },
                'semantic': {
                    'metric': 'euclidean',
                    'pooling': 'min'
                }
            }
        }
        yaml.dump(meta_dict, open(os.path.join(path, 'meta.yaml'), 'w'))

        print('Submission folder for epoch {} created'.format(epoch))


@hydra.main(config_path='../configs', config_name='config')
def eval(args):
    if args.quantized:
        clustering = None
        args.n_clusters += 1
    else:
        clustering = load_clustering(args)

    if args.arch == 'LSTM':
        model = LSTM_model(args.n_clusters, args)
    elif args.arch == 'QRNN':
        model = QRNN_model(args.n_clusters, args)
    else:
        raise ValueError('Architecture not valid')
    model.to(args.device)
    
    resume_epoch = 0
    model, _, resume_epoch = load_checkpoint(model, args)

    if args.nGPU > 1:
        model = torch.nn.DataParallel(model)
    print(model, '\n')

    eval_model(model, clustering, resume_epoch-1, args, dev_only=args.dev_only)


if __name__ == '__main__':
    eval()