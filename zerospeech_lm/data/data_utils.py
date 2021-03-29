import torch


def data_batchifier(data, args):
    # convert data to one vector with phones (ids) only
    batch_data = []
    ids = torch.randperm(len(data))
    for idx in ids:
        batch_data += data[idx]['id_data']
    batch_data = torch.tensor(batch_data)

    ## batchify
    nbatch = batch_data.size(0) // (args.bsz * args.seq_len)
    # trim extra elements that don't fit into last batch
    batch_data = batch_data[:nbatch*args.bsz*args.seq_len]
    # reshape into (nbatch, bsz, seq_len) so that batches split sequences
    batch_data = batch_data.view(args.bsz, -1).T.contiguous().view(nbatch, args.seq_len, args.bsz)

    return batch_data


def chr_to_id(seq, vocab):
    ids = []
    for ltr in seq:
        ids.append(vocab[ltr])
    return ids


def id_to_chr(seq, reversed_vocab):
    chars = ''
    for idx in seq:
        chars += reversed_vocab[idx]
    return chars