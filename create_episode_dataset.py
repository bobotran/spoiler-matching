import os
import json
from tqdm import tqdm
import random
from util import get_show_as_dataset
import numpy as np

def _get_summaries_as_dataset(inpath, show_name):
    '''Returns summaries as list [[show_name, episode_number, summary],]'''
    out = []
    with open(inpath, 'r') as f:
        summaries = json.load(f)
    for episode_num, summary in summaries.items():
        out.append((show_name, int(episode_num), summary))
    return out

def extract_summaries(shows, out_dir):
    all_summaries = []
    for fp in tqdm(shows):
        # Makes assumption that fp is .../show_name/summaries.json
        show_name = os.path.dirname(fp).split('/')[-1]
        if '::' in show_name:
            # Chop off season number
            show_name = show_name.split('::')[0]
        all_summaries += _get_summaries_as_dataset(fp, show_name)
    with open(os.path.join(out_dir, 'summaries.json'), 'w', encoding='utf-8') as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=4)

def _save_label_arr(out_file, arr):
    '''Saves label numpy array to json file
    Args:
        out_file: filepath to destination output json file
        arr: label numpy array to be saved to file
    '''
    print('Saving {} relevant comments with corresponding episode summaries to {}'.format(
        arr.shape[0], out_file))
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(arr[:, [2,3,1]].tolist(), f, ensure_ascii=False, indent=4)

def extract_labeled_dataset(shows, out_dir, split_ratio=-1):
    '''Args:
        split_ratio: If between 0 and 1, 
            then save split_ratio of the comments into split1.json 
            and (1 - split_ratio) of the comments into split2.json
    '''
    data = []
    for fp, to_exclude in tqdm(shows):
        # Makes assumption that fp is .../show_name/file.json
        show_name = os.path.dirname(fp).split('/')[-1]
        if '::' in show_name:
            # Chop off season number
            show_name = show_name.split('::')[0]
        data += get_show_as_dataset(fp, show_name, to_exclude, labeled=True)
    data = np.array(data, dtype=object)
    # Retain only relevant comments
    relevants = data[data[:, 0] == 0]

    RAND_SEED = 1
    rng = np.random.RandomState(RAND_SEED)
    rng.shuffle(relevants)

    if 0 < split_ratio and split_ratio < 1:
        num_datapoints = len(relevants)
        num_split = int(split_ratio * num_datapoints)

        split1, split2 = relevants[:num_split], relevants[num_split:]

        _save_label_arr(os.path.join(out_dir, 'split1.json'), split1)
        _save_label_arr(os.path.join(out_dir, 'split2.json'), split2)
    else:
        _save_label_arr(os.path.join(out_dir, 'out.json'), relevants)

def get_autolabeled_dataset(data_fp, logits_fp, out_dir, threshold=-1, summaries_fp=None, exclude_shows=[]):
    '''Args:
        data_fp: List of paths to .csv
        logits_fp: List of paths to .npy corresponding to data
        out_dir: Directory in which output train.json is saved
        threshold: Between 0 and 1. Used to classify examples after softmax.
            If unspecified, argmax is used to pick the class for each example.
        summaries_fp: Path to summaries.json. If specified, saves only messages
            that have the corresponding episode summary. 
    '''
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from torchmetrics.functional import auroc, fbeta_score, accuracy
    from torchmetrics.functional import precision as precision_score
    from torchmetrics.functional import recall as recall_score

    relevants_data = []
    num_chunks = len(data_fp)
    for i in range(num_chunks):
        data = pd.read_csv(data_fp[i], header=None).values

        logits = np.load(logits_fp[i])
        logits = torch.tensor(logits)
        scores = F.softmax(logits, dim=1)
        pos_score = scores[:, 1]
        if 0 < threshold and threshold < 1:
            autolabels = pos_score > threshold
        else:
            raise NotImplementedError()
        if data[0, 0] != -1:
            # Data is labeled; Print scores against autolabels for sanity check
            labels = torch.tensor(data[:, 0].astype(int))
            print('Threshold-based classifier:\n')
            print('AUROC: {}'.format(auroc(pos_score, labels, task='binary')))

            print('F1: {}'.format(fbeta_score(pos_score, labels, task='binary', beta=1.0, threshold=0.5, num_classes=1).item()))
            print('Precision: {}'.format(precision_score(pos_score, labels, task='binary', threshold=0.5, num_classes=1).item()))
            print('Recall: {}'.format(recall_score(pos_score, labels, task='binary', threshold=0.5, num_classes=1).item()))
            print('Accuracy: {}'.format(accuracy(pos_score, labels, task='binary', threshold=0.5, num_classes=1).item()))
            continue
    
        relevants_idx = autolabels.logical_not().nonzero(as_tuple=True)[0].numpy()

        if summaries_fp is not None:
            with open(summaries_fp) as f:
                summaries = json.load(f)
            summaries_keys = set()
            for (show, ep_num, _) in summaries:
                summaries_keys.add((show, ep_num))

            usable_relevants_idx = []
            for idx in relevants_idx:
                if (data[idx, 2], data[idx, 3]) in summaries_keys:
                    usable_relevants_idx.append(idx)
            relevants_idx = np.array(usable_relevants_idx)
        if len(exclude_shows) > 0:
            usable_relevants_idx = []
            for idx in relevants_idx:
                if data[idx, 2] not in exclude_shows:
                    usable_relevants_idx.append(idx)
            relevants_idx = np.array(usable_relevants_idx)
        relevants_data.append(data[relevants_idx])

    relevants_data = np.concatenate(relevants_data, axis=0)
    out_file = os.path.join(out_dir, 'train.json')
    print('Saving {} relevant comments with corresponding episode summaries to {}'.format(
        relevants_data.shape[0], out_file))
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(relevants_data[:, [2,3,1]].tolist(), f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    CHUNK_CSV_LIST = ['data/filtering/unlabeled/chunk_00.csv', 'data/filtering/unlabeled/chunk_01.csv']
    CHUNK_LOGITS_LIST = ['data/filtering/unlabeled/chunk_00_logits.npy', 'data/filtering/unlabeled/chunk_01_logits.npy']
    OUT_DIR = 'data/matching/with_autolabels'
    THRESHOLD = 0.42229151725769043
    SUMMARIES_FP = 'data/matching/handlabeled_only/summaries.json'

    get_autolabeled_dataset(
        CHUNK_CSV_LIST,
        CHUNK_LOGITS_LIST, 
        OUT_DIR,
        threshold=THRESHOLD,
        summaries_fp=SUMMARIES_FP,
        exclude_shows=[ # Exclude test set shows
            'arcane',
            'attack_on_titan',
            'kaguya_sama_love_is_war',
            'legend_of_korra'
        ])