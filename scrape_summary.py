from urllib.request import urlopen
from bs4 import BeautifulSoup
import bs4
import json
import os
from tqdm import tqdm

def is_caption(tag):
    '''Args:
        tag: bs4.element.Tag
    '''
    if 'class' in tag.attrs:
        return any(['caption' in c for c in tag.attrs['class']])
    return False

def get_summary(url, start_id):
    '''Scrapes webpage for `p` elements 
    starting at tag with id `start_id`
    and ending at the next element with the same heading level.
    '''
    html = urlopen(url).read()
    soup = BeautifulSoup(html, features="html.parser")

    start_tag = soup.find(id=start_id)
    assert start_tag is not None
    level = start_tag.parent.name
    assert 'h' in level

    text = []
    e = start_tag
    while not e.name == level:
        try: 
            if e.name == 'p' and not is_caption(e):
                text.append(e.get_text())
            e = e.next_element
        except Exception as e:
            print('Error on {}'.format(url))
            raise e
    text = ''.join(text).strip('\n').strip() # Strip trailing characters
    return " ".join(text.split()) # Remove duplicate whitespace

def scrape_summaries(summaries_list, start_id, outdir):
    '''Args:
        summaries_list: JSON filepath containing (episode number, summary url)
        start_id: HTML id of starting tag
    '''
    with open(summaries_list, 'r') as f:
        url_dict = json.load(f)

    summaries_dict = {}
    for episode_num in tqdm(url_dict):
        summaries_dict[episode_num] = get_summary(url_dict[episode_num], start_id)
        # summaries_dict[episode_num] += ' ' + get_summary(url_dict[episode_num], 'Plot')

    with open(os.path.join(outdir, 'summaries.json'), 'w', encoding='utf-8') as f:
        json.dump(summaries_dict, f, ensure_ascii=False, indent=4)