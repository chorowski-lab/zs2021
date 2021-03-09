import numpy 
from collections import defaultdict

class Dataset:
    def __init__(self, path, transform=None):
        self.data = []
        self.filenames = []
        self.n = 0
        self.transform = transform
        for line in open(path, 'r', encoding='utf8'):
            fname, sdesc = line.strip().split()
            self.filenames.append(fname)
            self.data.append(numpy.array(list(map(int, sdesc.split(','))), dtype='int32'))
            self.n += 1

    def __getitem__(self, i):
        if self.transform is not None:
            return self.filenames[i], self.transform(self.data[i])
        return self.filenames[i], self.data[i]

    def __len__(self):
        return self.n


class AlignedDataset(Dataset):
    def __init__(self, path, transform=None):
        super().__init__(path, transform)
        self.data = []
        self.filenames = ['BUNDLE']
        self.n = 1
        self.transform = transform
        for line in open(path, 'r', encoding='utf8'):
            fname, sdesc = line.strip().split()
            self.data.extend(list(map(int, sdesc.split(','))))
        self.data = numpy.array(self.data, dtype='int32')

    def __getitem__(self, i):
        return self.filenames[i], self.data


class AlignableTrainset(object):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.lls = defaultdict(int)
        for line in open(f'{path}/quantized_outputs.txt', 'r', encoding='utf8'):
            _, sdesc = line.strip().split()
            tokens = sdesc.split(',')
            ntokens = len(tokens)
            self.lls[ntokens] += 1
    
    def compute_batch_size(self, l):
        sz = 0
        for ll in self.lls:
            if ll >= l:
                sz += self.lls[ll] * (ll - l + 1)
        return sz
    
    def get_batch(self, length):
        bsz = self.compute_batch_size(length)
        batch = numpy.zeros((bsz, length))
        k = 0
        for line in open(f'{self.path}/quantized_outputs.txt', 'r', encoding='utf8'):
            _, sdesc = line.strip().split()
            tokens = numpy.array(list(map(int, sdesc.split(','))))
            if self.transform is not None:
                tokens = self.transform(tokens)
            for i in range(len(tokens) - length + 1):
                batch[k, :] = tokens[i:i+length]
                k += 1
        return batch

class AlignableTestset(object):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.data = defaultdict(list)
        self.fnames = defaultdict(list)
        for line in open(f'{path}/quantized_outputs.txt', 'r', encoding='utf8'):
            fname, sdesc = line.strip().split()
            tokens = numpy.array(list(map(int, sdesc.split(','))))
            ntokens = tokens.size
            self.data[ntokens].append(tokens)
            self.fnames[ntokens].append(fname)
    
    def lengths(self):
        return self.data.keys()
    
    def get_batch(self, length):
        return self.fnames[length], numpy.array(self.data[length])
    
    def __iter__(self):
        return self.data.items()

    def __len__(self):
        return len(self.data.keys())

