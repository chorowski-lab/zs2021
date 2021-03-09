class Dataset:
    def __init__(self, path, transform=None):
        self.data = []
        self.filenames = []
        self.n = 0
        self.transform = transform
        for line in open(path, 'r', encoding='utf8'):
            fname, sdesc = line.strip().split()
            self.filenames.append(fname)
            self.data.append(list(map(int, sdesc.split(','))))
            self.n += 1

    def __getitem__(self, i):
        if self.transform is not None:
            return self.filenames[i], self.transform(self.data[i])
        return self.filenames[i], self.data[i]

    def __len__(self):
        return self.n
