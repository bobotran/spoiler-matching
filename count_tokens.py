import json
from transformers import AutoTokenizer
import yaml

with open("data/episode_v1.1.0/summaries.json", 'r') as f:
    summaries = json.load(f)

# with open("tmp.yml", 'r') as f:
#     summaries = yaml.safe_load(f)

tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
# num_over = 0
# for summary in summaries:
#     encoding = tokenizer.encode(summary[-1])
#     # encoding = tokenizer.encode("I didn't like the transformation of Powder at the end. It was too quick too much of a turn for a character. Except her Fucking things up, nothing else was setup for her to turn the \"dark\" side. What reason does she even have to join with the guy who was going to kill her adoptee dad and her sister.\n\nSolid reason to hate her sister. Ok. But nothing suggests she would side Silco. Just left a bad test at the end for me. First two episodes were phenomenal though")
#     if len(encoding) > 3900:
#         num_over += 1
#     print(len(encoding))
# print('Number of summaries over 3900 tokens: {} / {}'.format(num_over, len(summaries)))

for (show_name, ep_num, summary) in summaries:
    if show_name == 'game_of_thrones' and ep_num == 25:
        print(len(tokenizer.encode(summary)))
        with open('tmp/tmp.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=4)
        break