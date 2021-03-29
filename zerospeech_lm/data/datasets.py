import os
import tqdm

import numpy as np
from torch.utils.data import Dataset, DataLoader


class CPC_feature_Dataset(Dataset):
    def __init__(self, args):
        self.data_dir = args.data_dir
        self.seq_alignment = args.seq_alignment
        self.file_names = []
        self.current_epoch_ids = None
        
        for file_name in os.listdir(self.data_dir):
            if file_name[-4:] != '.npy':
                continue
            self.file_names.append(file_name)
        print('Found {} files for CPC_feature_Dataset!\n'.format(len(self.file_names)))
    
    def get_batch(self, batch_n, n_batches, args):
        if self.seq_alignment:
            # padding and cropping
            batch_ids = self.current_epoch_ids[int(batch_n*args.bsz) : min(int((batch_n+1)*args.bsz), len(self.file_names))]
            batch_data = []
            # for idx in tqdm.tqdm(batch_ids, desc='Loading batch {}/{}'.format(batch_n, n_batches), ascii=True):
            for idx in batch_ids:
                datum = np.load(os.path.join(self.data_dir, self.file_names[idx]))
                if datum.shape[0] >= args.seq_len:
                    # crop right
                    batch_data.append(datum[:args.seq_len])
                elif datum.shape[0] < args.seq_len:
                    # pad right
                    datum = np.concatenate((datum, np.zeros((args.seq_len - datum.shape[0], datum.shape[1]))), axis=0)
                    batch_data.append(datum)
            
        else:
            # batchify
            pass
        return np.array(batch_data)
        
    def set_epoch_ids(self):
        self.current_epoch_ids = np.random.permutation(len(self.file_names))
    
    def create_vocab(self, args):
        pass

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        return np.load(os.path.join(self.data_dir, self.file_names[idx]))


class Quantized_Dataset(Dataset):
    def __init__(self, args):
        self.data_dir = os.path.join(args.data_dir, 'quantized_outputs.txt')
        self.current_epoch_ids = None
        
        with open(self.data_dir, 'r') as f:
            self.raw_data = f.readlines()
        
        self.data = []
        self.file_names = []
        for line in tqdm.tqdm(self.raw_data, desc='Loading Quantized_Dataset', ascii=True):
            t_idx = line.find('\t')
            self.file_names.append(line[:t_idx])
            str_data = line[t_idx+1:-1].split(',')
            if str_data[-1] == '':
                str_data = str_data[:-1]
            self.data.append(np.array(str_data, dtype=np.int32))
        print('\n')
            
    def get_batch(self, batch_n, n_batches, args):
        batch_ids = self.current_epoch_ids[int(batch_n*args.bsz) : min(int((batch_n+1)*args.bsz), len(self.data))]
        batch_data = []
        for idx in batch_ids:
            datum = self.data[idx]
            if datum.shape[0] >= args.seq_len:
                # crop right
                batch_data.append(datum[:args.seq_len])
            elif datum.shape[0] < args.seq_len:
                # pad right
                datum = np.concatenate((datum, np.zeros((args.seq_len - datum.shape[0])) + args.n_clusters), axis=0)
                batch_data.append(datum)
        return np.array(batch_data)

    def set_epoch_ids(self):
        self.current_epoch_ids = np.random.permutation(len(self.data))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.file_names[idx]