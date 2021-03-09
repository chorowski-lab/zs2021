import argparse

parser = argparse.ArgumentParser(description='Compute the pseudo log-proba of a list of sentences')

parser.add_argument('data', type=str,
                    help='Path to the folder containing the "dev-i" files')

parser.add_argument('--method', type=str, default='min',
                    help='Method for evaluating pseudo logprob. Available: "min", "sum". Defaults to "min"')
parser.add_argument('--n', type=int, default=20,
                    help='Number of dev-i files. Default: 20')

args = parser.parse_args()

def _min(desc):
    d = desc[0][1:-1]
    r = {int(x.split(':')[0]): int(x.split(':')[1]) for x in d.split(',') }
    return - min(r.keys())

def _sum(desc):
    d = desc[0][1:-1]
    r = {int(x.split(':')[0]): int(x.split(':')[1]) for x in d.split(',') }
    return - sum(k * v for k, v in sorted(r.items()))

method = None

if args.method == 'min':
    method = _min

if args.method == 'sum':
    method = _sum

if method is None:
    raise 'Invalid "method" parameter'

with open('./output/lexical/dev.txt', 'w', encoding='utf8') as out:
    for i in range(1, args.n+1):
        for line in open(f'{args.data}/dev-{i}', 'r'):
            fname = line.strip().split()[0]
            desc = line.strip().split()[1:]
            out.write(f'{fname} {method(desc)}\n')