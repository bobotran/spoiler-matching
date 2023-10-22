import os
import json
from tqdm import tqdm

VERBOSE = True

def load_by_episode(json_path):
    '''Loads json list of messages and returns as dict
    where key is episode, value is comments.
    '''
    with open(json_path, 'r') as f:
        comments = json.load(f)
    comments_by_episode = {}
    for c in comments:
        if c['episode'] in comments_by_episode:
            comments_by_episode[c['episode']].append(c)
        else:
            comments_by_episode[c['episode']] = [c]
    return comments_by_episode

from markdown import Markdown
from io import StringIO

def _unmark_element(element, stream=None):
    if stream is None:
        stream = StringIO()
    if element.text:
        stream.write(element.text)
    for sub in element:
        _unmark_element(sub, stream)
    if element.tail:
        stream.write(element.tail)
    return stream.getvalue()


# patching Markdown
Markdown.output_formats["plain"] = _unmark_element
__md = Markdown(output_format="plain")
__md.stripTopLevelTags = False


def unmark(text):
    return __md.convert(text)

import re

def _unwrap_pattern(pattern, s, pre_length, post_length):
    '''Replaces instances of pattern in s with the text inside patter
    Args:
        pre_length: Number of characters at the beginning of the pattern
        post_length: Number of characters at the end of the pattern
    '''
    matches = re.findall(pattern, s)
    
    cleaned = s
    for match in matches:
        cleaned = cleaned.replace(match, match[pre_length:-post_length], 1)
    return cleaned

def _unwrap(s):
    s = _unwrap_pattern('~{2}[^~{2}]*~{2}', s, 2, 2)
    s = _unwrap_pattern('>![^!]*!<', s, 2, 2)
    s = _unwrap_pattern('\^\([^\)]*\)', s, 2, 1)
    s = s.replace('^', '')
    return s

def _clean_message(message):
    s = message.replace('] (', '](') # Convert Reddit-style links into standard markdown
    while True: # Peel back layers of markdown
        cleaned = unmark(s)
        if s == cleaned:
            break
        s = cleaned

    s = _unwrap(s)
    # Link removal
    for link in re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', s):
        if link[-1] == ')':
            link = '(' + link
        if link[-1] == ']':
            link = '[' + link
        s = s.replace(link, '')

    s = unmark(s) # One last unmark in case link removal left markdown elements
    s = " ".join(s.split()) # Remove duplicate whitespace
    if s.lower() == 'na':
        s = '' # Strip these messages as they get converted into NANs unintentionally by Pandas
    return s

def _get_messages_as_data(messages, show_name, exclude_episodes=[], label_field='label'):
    if VERBOSE:
        print('{} total messages: {}'.format(show_name, len(messages)))
    exclude_episodes = set(exclude_episodes)

    if len(label_field) > 0:
        filtered = [(1 if 'Irrelevant' in m[label_field] else 0, 
            _clean_message(m['data']), 
            show_name,
            int(m['episode'])) 
            for m in messages if 
            len(m[label_field]) > 0 and
            'Uncertain' not in m[label_field] and 
            'Invalid' not in m[label_field] and
            int(m['episode']) not in exclude_episodes]
    else:
        # Unlabeled data
        def _get_datapoint(m):
            return (-1, _clean_message(m['text']), show_name, int(m['episode']))
        filtered = Parallel(n_jobs=6)(delayed(_get_datapoint)(m) for m in messages if 
            int(m['episode']) not in exclude_episodes)

    filtered = [d for d in filtered if len(d[1]) > 0]

    if VERBOSE:
        print('{} filtered messages: {}'.format(show_name, len(filtered)))
    return filtered

from joblib import Parallel, delayed

def get_show_as_dataset(inpath, show_name, exclude_episodes=[], labeled=True):
    '''Reads in a labeled dataset in doccano format.
    Filters out "invalid" and "uncertain" examples.
    Args:
        exclude_episodes: If specified, throws away messages from these episodes.
    '''
    with open(inpath, 'r') as f:
        data = json.load(f)
    return _get_messages_as_data(data, show_name, 
        exclude_episodes=exclude_episodes, label_field='label' if labeled else '')