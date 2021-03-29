import os
import hydra
from omegaconf import DictConfig, OmegaConf
import yaml
from tqdm import tqdm
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from zerospeech_lm.data.datasets import CPC_feature_Dataset, Quantized_Dataset
from zerospeech_lm.data.data_utils import data_batchifier

from zerospeech_lm.models.lm import LSTM_model, QRNN_model
from zerospeech_lm.models.utils import save_checkpoint, load_checkpoint, reset_hidden
from zerospeech_lm.models.clustering import train_clustering, load_clustering

from zerospeech_lm.scripts.evaluation import eval_model


def train_model(epoch, model, optimizer, criterion, clustering, data, writer, global_iter, args):
    loss_acc = 0.
    n_ltr = 0.
    score = 0.

    data.set_epoch_ids()
    n_batches = len(data) // args.bsz
    pbar = tqdm(range(n_batches), desc='Epoch ' + str(epoch), ascii=True)
    for i in pbar:
        batch = data.get_batch(i, n_batches, args)
        if clustering is None:
            clustered_batch = batch
        else:
            clustered_batch = clustering.predict(batch.reshape(-1, batch.shape[-1])).reshape(args.bsz, args.seq_len)
        clustered_batch = torch.from_numpy(clustered_batch).to(torch.int64)

        if args.inverse_seqs:
            clustered_batch = clustered_batch.flip(1) # flip sequences for backward training
            
        x = clustered_batch[:,:-1].to(args.device)
        y = clustered_batch[:,1:].to(args.device)
        
        hidden_state = reset_hidden(args)
        probs, _ = model(x, hidden_state)
        preds = torch.argmax(probs, dim=2)

        optimizer.zero_grad()
        loss = criterion(probs.permute(0, 2, 1), y)
        loss.backward()
        optimizer.step()

        n_ltr += clustered_batch.size(0) * clustered_batch.size(1)
        score += torch.sum(y.flatten() == preds.flatten()).item()
        loss_acc += loss.item()

        writer.add_scalar("Accuracy", score/n_ltr*100, global_step=global_iter)
        writer.add_scalar("Perplexity", np.exp(loss_acc / (i+1)), global_step=global_iter)

        pbar.set_postfix(
            OrderedDict({
                'Train accuracy': "%.4f" % (score/n_ltr*100),
                'Train perplexity': "%.4f" % np.exp(loss_acc / (i+1))
            })
        )
        global_iter += 1
    
    return model, optimizer, global_iter

@hydra.main(config_path='../configs', config_name='config')
def train(args):
    if args.quantized:
        data = Quantized_Dataset(args)
        n_clusters = args.n_clusters + 1 # additional cluster for blank token used in padding
        clustering = None
    else:
        data = CPC_feature_Dataset(args)
        if args.load_clustering:
            clustering = load_clustering(args)
        else:
            clustering = train_clustering(data, args)

    if args.arch == 'LSTM':
        model = LSTM_model(n_clusters, args)
    elif args.arch == 'QRNN':
        model = QRNN_model(n_clusters, args)
    else:
        raise ValueError('Architecture not valid')
    model.to(args.device)
    
    resume_epoch = 0
    if args.load_model:
        model, optimizer, resume_epoch = load_checkpoint(model, args)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.nGPU > 1:
        model = torch.nn.DataParallel(model)
    print(model, '\n')

    total = 0
    for p in model.parameters():
        if p.requires_grad:
            total += torch.prod(torch.tensor(p.size()))
    print('Total number of trainable parameters: {} \n \n'.format(total.item()))

    criterion = nn.CrossEntropyLoss()

    global_iter = 0
    train_writer = SummaryWriter(os.path.join(args.tensorboard_dir, 'train'))
    for epoch in range(resume_epoch, args.num_epochs):
        model.train()
        model, optimizer, global_iter = train_model(epoch, model, optimizer, criterion, clustering, data, train_writer, global_iter, args)
        save_checkpoint(model, optimizer, epoch, args)
        # if epoch % 2 == 0:
        #     eval_model(model, clustering, epoch, args)


if __name__ == "__main__":
    train()