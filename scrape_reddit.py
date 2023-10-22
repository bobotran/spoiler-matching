import praw
import pprint
import json
import random
import os
import util
from tqdm import tqdm

reddit = praw.Reddit("bot1")

def _get_episode_comments(ep_num, ep_link):
    submission = reddit.submission(url=ep_link)
    submission.comment_sort = 'new'
    submission.comments.replace_more(limit=None)
    datalist = []
    for top_level_comment in submission.comments:
        if top_level_comment is None or \
            top_level_comment.author is None or \
            top_level_comment.removal_reason is not None or \
            top_level_comment.report_reasons is not None or \
            'bot' in top_level_comment.author.name.lower():
            continue
        datalist.append({
            'episode': ep_num,
            'id': top_level_comment.permalink, 
            'text': top_level_comment.body,
            'label': ['Relevant']})
    return datalist

def scrape_all_comments(episodes_list, outpath):
    '''Scrapes top-level comments from each episode in episodes_list
    and saves list to outpath
    Args:
        episodes_list: Path to json where key is episode number,
            value is reddit thread link
    '''
    with open(episodes_list, 'r') as f:
        episodes_dict = json.load(f)

    data = []
    for ep_num, ep_link in tqdm(episodes_dict.items()):
        if isinstance(ep_link, list):
            for epl in ep_link:
                data += _get_episode_comments(ep_num, epl)
        elif isinstance(ep_link, str):
            data += _get_episode_comments(ep_num, ep_link)
        else:
            raise ValueError()
    print('Top-level comments: {}'.format(len(data)))
    # random.Random(1).shuffle(datalist)
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def split_data(inpath, outdir, num_to_label=50):
    '''Split data into set to be labeled and set to be auto-labeled
    Args:
        inpath: Path to json with all top-level comments
        outdir: Directory to save to_label.json and unlabeled.json
    '''
    all_comments = util.load_by_episode(inpath)
    to_label_path = os.path.join(outdir, 'to_label.json')

    def split_episode(pool, to_label_ep=[], num_to_label=50):
        '''If to_label_ep exists, split new messages in pool to unlabeled_ep.
        If not, select num_to_label random messages from to be to_label_ep.
        '''
        if len(to_label_ep) > 0:
            labeled_ids = [c['id'] for c in to_label_ep]
            unlabeled_ep = [c for c in pool if c['id'] not in labeled_ids]
            # Add more comments from unlabeled pool into the set to be labeled
            to_add = num_to_label - len(to_label_ep)
            if to_add > 0:
                random.Random(1).shuffle(unlabeled_ep)
                to_label_ep += unlabeled_ep[:to_add]
                unlabeled_ep = unlabeled_ep[to_add:]
        else:
            random.Random(1).shuffle(pool)
            to_label_ep = pool[:num_to_label]
            unlabeled_ep = pool[num_to_label:]
        return to_label_ep, unlabeled_ep

    unlabeled = []
    to_label = []
    # Keep label set the same by assigning new comments to unlabeled
    if os.path.exists(to_label_path):
        print('Found {}! Preserving those labels.'.format(to_label_path))
        to_label_og = util.load_by_episode(to_label_path)
    else:
        to_label_og = {}

    for episode in all_comments:
        to_label_ep, unlabeled_ep = split_episode(all_comments[episode], 
            to_label_og[episode] if episode in to_label_og else [], 
            num_to_label=num_to_label)
        to_label += to_label_ep
        unlabeled += unlabeled_ep
            
    with open(os.path.join(outdir, 'to_label.json'), 'w', encoding='utf-8') as f:
        json.dump(to_label, f, ensure_ascii=False, indent=4)

    with open(os.path.join(outdir, 'unlabeled.json'), 'w', encoding='utf-8') as f:
        json.dump(unlabeled, f, ensure_ascii=False, indent=4)

    print('To Label: {}'.format(len(to_label)))
    print('Unlabeled: {}'.format(len(unlabeled)))
    