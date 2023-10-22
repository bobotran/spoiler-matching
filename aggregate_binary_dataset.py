'''Aggregates output JSON list from scrape_reddit.py from multiple shows to Relevant/Irrelevant dataset format
    by filtering out Invalids and cleaning messages of special characters'''
import os
from tqdm import tqdm
import random
from util import get_show_as_dataset
import random
import pandas as pd
import os

def extract_labeled_dataset(shows, out_dir, prefix=''):
    '''Converts JSON data to dataset format.
    Irrelevant is labeled as 1, Relevant as 0.
    All other labels are discarded.
    Args:
        paths: [(json_path_to_data, [episodes_to_exclude]),]
        out_dir: Directory in which train.csv and test.csv are saved
    '''
    data = []
    for fp, to_exclude in tqdm(shows):
        # Makes assumption that fp is .../show_name/file.json
        show_name = os.path.dirname(fp).split('/')[-1]
        if '::' in show_name:
            # Chop off season number
            show_name = show_name.split('::')[0]
        data += get_show_as_dataset(fp, show_name, to_exclude, labeled=True)

    RAND_SEED = 1

    num_datapoints = len(data)
    rng = random.Random(RAND_SEED)
    rng.shuffle(data)

    test_split = int(0.90 * num_datapoints) # 70 - 20 - 10 split
    trainset, testset = data[:test_split], data[test_split:]

    pd.DataFrame(trainset).to_csv(os.path.join(out_dir, prefix + 'train.csv'), index=False, header=False)
    pd.DataFrame(testset).to_csv(os.path.join(out_dir, prefix + 'test.csv'), index=False, header=False)

    print('Total Labeled Messages: {}'.format(num_datapoints))

    num_relevant = sum([d[0] == 0 for d in trainset])
    num_irrelevant = sum([d[0] == 1 for d in trainset])
    print('Number of Relevant examples: {} / {}'.format(num_relevant, len(trainset)))
    print('Set K = {} for 70-20 train-val split'.format(int((7/9) * min(num_relevant, num_irrelevant))))

def extract_unlabeled_dataset(shows, out_dir, prefix=''):
    '''Converts JSON data to dataset format.
    Irrelevant is labeled as 1, Relevant as 0. To be labeled as -1.
    All other labels are discarded.
    Args:
        paths: [(json_path_to_data, [episodes_to_exclude]),]
        out_dir: Directory in which train.csv and test.csv are saved
    '''
    data = []
    for fp, to_exclude in tqdm(shows):
        # Makes assumption that fp is .../show_name/file.json
        show_name = os.path.dirname(fp).split('/')[-1]
        if '::' in show_name:
            # Chop off season number
            show_name = show_name.split('::')[0]
        data += get_show_as_dataset(fp, show_name, to_exclude, labeled=False)

    num_datapoints = len(data)

    # Copy train.csv from labeled set
    pd.DataFrame(data).to_csv(os.path.join(out_dir, prefix + 'test.csv'), index=False, header=False)

    print('Total Unlabeled Messages: {}'.format(num_datapoints))

def split_csv(original, chunk_size):
    '''Splits original csv into chunks of size chunk_size and saves them to out_dir'''
    df = pd.read_csv(original, header=None)
    idx = 0
    while df.shape[0] > 0:
        chunk = df.iloc[:chunk_size,:]
        chunk.to_csv(os.path.join(os.path.dirname(original), 'chunk_{:02d}.csv'.format(idx)), index=False, header=False)
        df = df.iloc[chunk_size:,:]
        idx += 1