import argparse
from os import listdir
import sys

def parseArgs():
    parser = argparse.ArgumentParser(description='Compute the pseudo log-proba of a list of sentences')

    parser.add_argument('data', type=str,
                        help='Path to the folder containing the "dev-i" files')

    parser.add_argument('--method', type=str, default='min',
                        help='Method for evaluating pseudo logprob. Available: "min", "sum", "firstk". Defaults to "min"')

    parser.add_argument('--k', type=int,
                        help='Parameter for the "first_k" method.')
    parser.add_argument('--n', type=int, 
                        help='Number of the dev-i files. Default: 20')
    parser.add_argument('--q', type=int, 
                        help='Parameter for the "min_x" method.')

    return parser.parse_args()

def _min(desc):
    d = desc[0][1:-1]
    r = {int(x.split(':')[0]): int(x.split(':')[1]) for x in d.split(',') }
    total = sum(k * v for k, v in sorted(r.items()))
    return - min(r.keys()) / total

def _test(desc):
    d = desc[0][1:-1]
    r = {int(x.split(':')[0]): int(x.split(':')[1]) for x in d.split(',') }
    return - min(r.keys()) - sum(k * v for k, v in sorted(r.items())) / 100000000000000000

def _firstk(k):
    def f(desc):
        q = k
        d = desc[0][1:-1]
        r = {int(x.split(':')[0]): int(x.split(':')[1]) for x in d.split(',') }
        s = 0
        w = 0
        for c, v in sorted(r.items()):
            s -= min(q, v) * c * (2 ** w)
            q -= v
            w += 1
            if q <= 0:
                break
        return s
    return f

def _sum(desc):
    d = desc[0][1:-1]
    r = {int(x.split(':')[0]): int(x.split(':')[1]) for x in d.split(',') }
    return - sum(k * v for k, v in sorted(r.items()))

def _x_min(args, dtoi):
    if args.q is None:
        def f(desc):
            r = desc[1:-1].split(',')
            return - float(r[0])
    else:
        k = dtoi[args.q] if dtoi is not None else args.q
        def f(desc):
            r = desc[1:-1].split(',')
            return - float(r[0]) / float(r[k])
    return f


def main(args):
    try:
        variant = None

        for line in open(f'{args.data}/dev-1', 'r'):
            desc = line.strip().split()
            if len(desc[1:]) == 2:
                variant = 'dtw-x'
            elif len(desc[1:]) > 1:
                variant = 'lookup'
            elif desc[1][0] == '{' and desc[1][-1] == '}':
                variant = 'editdist/dtw'
            break

        if variant is None:
            raise ValueError(f'unrecognized format of dev-i files')
        
        print(f'Recognized variant: {variant}')

        if args.n is None:
            ids = [int(fname.split('-')[1]) for fname in listdir(args.data) if fname.startswith('dev-')]
            n = max(ids)

            if len(set(range(1, n+1)) - set(ids)) > 0:
                raise ValueError(f'some dev-i files are missing')
        else:
            n = args.n
            ids = [int(fname.split('-')[1]) for fname in listdir(args.data) if fname.startswith('dev-')]

            if len(set(range(1, n+1)) - set(ids)) > 0:
                raise ValueError(f'some dev-i files are missing')

        if variant == 'editdist/dtw':
            method = None

            if args.method == 'min':
                method = _min

            if args.method == 'sum':
                method = _sum
            
            if args.method == 'firstk':
                method = _firstk(args.k)

            if method is None:
                raise ValueError('invalid "method" parameter, must be either sum, min or firstk')

            print(f'Using method: {args.method}')

            with open('./output/lexical/dev.txt', 'w', encoding='utf8') as out:
                for i in range(1, n+1):
                    for line in open(f'{args.data}/dev-{i}', 'r'):
                        fname = line.strip().split()[0]
                        desc = line.strip().split()[1:]
                        out.write(f'{fname} {method(desc)}\n')

        elif variant == 'lookup':
            with open('./output/lexical/dev.txt', 'w', encoding='utf8') as out:
                for i in range(1, n+1):
                    for line in open(f'{args.data}/dev-{i}'):
                        fname = line.split()[0]
                        score = 0
                        for s in line.split()[1:]:
                            score = score / 100 + int(s)
                        out.write(f'{fname} {score}\n')

        elif variant == 'dtw-x':
            
            dtoi = None
            if '!info.txt' in listdir(args.data):
                print('Found !info.txt file, loading')
                dtoi = dict()
                i = 0
                for line in open(f'{args.data}/!info.txt', 'r'):
                    if line.strip().find('-') != -1:
                        a, b = map(int, line.strip().split('-'))
                        for q in range(a, b+1):
                            dtoi[q] = i
                            i += 1
                    else:
                        dtoi[int(line.strip())] = i
                        i += 1

            method = None

            if args.method == 'min':
                method = _x_min(args, dtoi)
            
            if method is None:
                raise ValueError('invalid "method" parameter, must be min')

            print(f'Using method: {args.method}')

            with open('./output/lexical/dev.txt', 'w', encoding='utf8') as out: 
                for i in range(1, n+1):
                    for line in open(f'{args.data}/dev-{i}', 'r'):
                        fname = line.strip().split()[0]
                        desc = line.strip().split()[1]
                        out.write(f'{fname} {method(desc)}\n')

        print('Done!')
    except ValueError as error:
        print(f'ERROR: {error}')
        sys.exit(-1)


if __name__ == "__main__":
    args = parseArgs()
    main(args)