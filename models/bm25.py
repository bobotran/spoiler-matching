import torchmetrics
import torch
import torch.nn.functional as F
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi

def get_bm25_scores(summaries, comment, stem=True, remove_stopwords=True):
    '''Args:
        summaries: List[str]
        comment: str
    '''
    from nltk.corpus import stopwords

    tokenized_summaries = [summary.lower().split(" ") for summary in summaries]
    tokenized_comment = comment.lower().split(" ")
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokenized_summaries = [[w for w in summary if w not in stop_words] for summary in tokenized_summaries]
        tokenized_comment = [w for w in tokenized_comment if w not in stop_words]
    if stem:
        ps = PorterStemmer()
        tokenized_summaries = [[ps.stem(w) for w in summary] for summary in tokenized_summaries]
        tokenized_comment = [ps.stem(w) for w in tokenized_comment]
    bm25 = BM25Okapi(tokenized_summaries)
    return bm25.get_scores(tokenized_comment)

if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import argparse
    import json
    from tqdm import tqdm
    import numpy as np
    from joblib import Parallel, delayed

    parser = argparse.ArgumentParser()
    parser.add_argument('--comments', required=True)
    args = parser.parse_args()

    with open('data/episode_v1.1.0/summaries.json') as f:
        summaries_raw = json.load(f)
    summaries = {}
    for (show_name, ep_num, summary) in summaries_raw:
        summaries[(show_name, ep_num)] = summary

    with open(args.comments) as f:
        comments = json.load(f)

    ps = PorterStemmer()

    print('{} comments'.format(len(comments)))
    def _get_bm25_score(target_show_name, target_ep_num, comment):
        pertinent_summaries = [(show_name, ep_num) for (show_name, ep_num) in summaries if show_name == target_show_name]

        targets = [(True if (show_name, ep_num) == (target_show_name, target_ep_num) else False) \
            for (show_name, ep_num) in pertinent_summaries]
        pertinent_summaries = [summaries[(show_name, ep_num)] for (show_name, ep_num) in pertinent_summaries]
        
        bm25_scores = get_bm25_scores(pertinent_summaries, comment)
        return torchmetrics.functional.retrieval_reciprocal_rank(torch.tensor(bm25_scores), torch.tensor(targets)).item()
    rr = Parallel(n_jobs=8, verbose=5)(delayed(_get_bm25_score)(name, num, comment) for (name, num, comment) in comments)
    # rr = []
    # for (target_show_name, target_ep_num, comment) in tqdm(comments):
    #     rr.append(_get_bm25_score(target_show_name, target_ep_num, comment))
    print('MRR: {}'.format(np.mean(rr)))
