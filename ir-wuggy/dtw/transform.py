import argparse
import sys
import pathlib
from more_itertools import pairwise

def parseArgs():
    parser = argparse.ArgumentParser(description='Apply the transformation to the quantized data set')
    parser.add_argument('dataset', type=pathlib.Path,
                        help='Location of the dataset file (quantized_outputs.txt)')
    parser.add_argument('output', type=pathlib.Path,
                        help='Location of the output file')
    parser.add_argument('--transform', type=str, default='squash',
                        help='Transformation to be used, available: \'squash\'. Default: \'squash\'')
    return parser.parse_args()




def squash(data):
    return [x1 for x1, x2 in pairwise(data) if x1 != x2] + data[-1:]


def main(args):
    try:
        transform = None

        if args.transform == 'squash':
            transform = squash

        if transform is None:
            raise ValueError('invalid transform')

        with open(args.output, 'w', encoding='utf8') as out:
            for line in open(args.dataset, 'r', encoding='utf8'):
                fname, desc = line.strip().split()
                data = list(map(int, desc.split(',')))
                out.write(f'{fname} {",".join(map(str, transform(data)))}\n')

        print('DONE!')

    except ValueError as error:
        print(f'ERROR: {error}')
        sys.exit(-1)

if __name__ == "__main__":
    args = parseArgs()
    main(args)