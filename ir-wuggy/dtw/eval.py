import argparse
import sys
import pathlib
import pandas

def parseArgs():
    parser = argparse.ArgumentParser(description='Compute the pseudo log-proba of a list of sentences')
    parser.add_argument('--submission', type=pathlib.Path, default=pathlib.Path('./output/lexical/dev.txt'),
                        help='Location of the dev.txt file')
    parser.add_argument('--gold', type=pathlib.Path, default=pathlib.Path('/pio/data/zerospeech2021/dataset/lexical/dev/gold.csv'),
                        help='Location of the gold.csv file')
    
    return parser.parse_args()


def load_data(gold_file, submission_file):
    # ensures the two input files are here
    for input_file in (gold_file, submission_file):
        if not pathlib.Path(input_file).is_file():
            raise ValueError(f'file not found: {input_file}')

    # load them as data frames indexed by filenames
    gold = pandas.read_csv(
        gold_file, header=0, index_col='filename').astype(
            {'frequency': pandas.Int64Dtype()})
    score = pandas.read_csv(
        submission_file, sep=' ', header=None,
        names=['filename', 'score'], index_col='filename')
    
    # merge the gold and score using filenames, then remove the columns
    # 'phones' and 'filename' as we don't use them for evaluation
    data = pandas.concat([gold, score], axis=1)
    data.reset_index(inplace=True)
    data.drop(columns=['phones', 'filename'], inplace=True)
    # going from a word per line to a pair (word, non word) per line
    data = pandas.concat([
        data.loc[data['correct'] == 1].reset_index().rename(
            lambda x: 'w_' + x, axis=1),
        data.loc[data['correct'] == 0].reset_index().rename(
            lambda x: 'nw_' + x, axis=1)], axis=1)
    data.drop(
        ['w_index', 'nw_index', 'nw_voice', 'nw_frequency',
         'w_correct', 'nw_correct', 'nw_id', 'nw_length'],
        axis=1, inplace=True)
    data = data[data['w_score'].notna()]
    data = data[data['nw_score'].notna()]
    data.rename(
        {'w_id': 'id', 'w_voice': 'voice', 'w_frequency': 'frequency',
         'w_word': 'word', 'nw_word': 'non word', 'w_length': 'length',
         'w_score': 'score word', 'nw_score': 'score non word'},
        axis=1, inplace=True)

    return data


def evaluate_by_pair(data):
    # compute the score for each pair in an additional 'score' column, then
    # delete the 'score word' and 'score non word' columns that become useless
    score = data.loc[:, ['score word', 'score non word']].to_numpy()
    data['score'] = (
        0.5 * (score[:, 0] == score[:, 1])
        + (score[:, 0] > score[:, 1]))
    data.drop(columns=['score word', 'score non word'], inplace=True)

    # finally get the mean score across voices for all pairs
    score = data.groupby('id').apply(lambda x: (
        x.iat[0, 3],  # word
        x.iat[0, 5],  # non word
        x.iat[0, 2],  # frequency
        x.iat[0, 4],  # length
        x['score'].mean()))
    return pandas.DataFrame(
        score.to_list(),
        columns=['word', 'non word', 'frequency', 'length', 'score'])


def evaluate_by_frequency(by_pair):
    bands = pandas.cut(
        by_pair.frequency,
        [0, 1, 5, 20, 100, float('inf')],
        labels=['oov', '1-5', '6-20', '21-100', '>100'],
        right=False)

    return by_pair.score.groupby(bands).agg(
        n='count', score='mean', std='std').reset_index()


def evaluate_by_length(by_pair):
    return by_pair.score.groupby(by_pair.length).agg(
        n='count', score='mean', std='std').reset_index()


def write_csv(frame, filename):
    frame.to_csv(filename, index=False, float_format='%.4f')
    print(f'  > Wrote {filename}')


def main(args):
    try:
        output = pathlib.Path('.')

        gold_file = args.gold
        submission_file = args.submission

        data = load_data(gold_file, submission_file)

        by_pair = evaluate_by_pair(data)
        by_frequency = evaluate_by_frequency(by_pair)
        by_length = evaluate_by_length(by_pair)

        print(f'Lexical score: {by_pair.score.mean() * 100}')
        print(f'No oov score:  {by_pair[by_pair.frequency != 0].score.mean() * 100}')

        by_pair.drop(['frequency', 'length'], axis=1, inplace=True)

        write_csv(by_pair, output / f'score_lexical_dev_by_pair.csv')
        write_csv(by_frequency, output / f'score_lexical_dev_by_frequency.csv')
        write_csv(by_length, output / f'score_lexical_dev_by_length.csv')


    except ValueError as error:
        print(f'ERROR: {error}')
        sys.exit(-1)

    

if __name__ == "__main__":
    args = parseArgs()
    main(args)