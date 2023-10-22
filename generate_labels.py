import os
import json
from tqdm import tqdm
import random
from collections import OrderedDict

def get_label(label_name, prefixKey=None, suffixKey=None, backgroundColor=None, textColor='#ffffff'):
    '''Returns OrderedDict object with fields
    id, text, prefixKey, suffixKey, backgroundColor, textColor.
    '''
    if backgroundColor is None:
        backgroundColor = '#%06x' % random.randint(0, 0xFFFFFF)
    return OrderedDict([#('id', str(id)),
                ('text', str(label_name)),
                ('prefixKey', prefixKey),
                ('suffixKey', suffixKey),
                ('backgroundColor', backgroundColor),
                ('textColor', textColor)])

def get_labels(episode_range):
    '''Returns list of OrderedDict objects each with fields
    id, text, prefixKey, suffixKey, backgroundColor, textColor.
    First two labels are always "Invalid" and "Non-spoiler"
    '''
    labels = [# get_label('Invalid', backgroundColor='#4287f5'), 
            get_label('Irrelevant', backgroundColor='#4287f5'),
            get_label('Relevant', backgroundColor='#218f33'),
            # get_label('Future', backgroundColor='#eb4034'),
            get_label('Uncertain', backgroundColor='#F321CC '),
            ]
    # for i in range(episode_range[0], episode_range[1]+1):
    #     labels.append(get_label(i))
    return labels

if __name__ == '__main__':
    EPISODE_RANGE = (1, 6)
    DIR = 'data/spy_x_family/'

    labels = get_labels(EPISODE_RANGE)
    fp = os.path.join(DIR, 'label_names.json')
    with open(fp, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=4)