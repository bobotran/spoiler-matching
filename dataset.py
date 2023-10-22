import numpy as np
import pytorch_lightning as pl
import torch
import json
from transformers import AutoTokenizer, AutoConfig
import random
import pickle
from models.bm25 import get_bm25_scores
import nltk
nltk.download('punkt')

class BaseRelevantEpisodeSet(torch.utils.data.Dataset):
    '''Abstract base class for RelevantEpisodeTrainSet and RelevantEpisodeTestSet'''
    def __init__(self, summaries_fp, comments_fp, shows=set(), chunk_size=-1):
        '''For each comment, returns its concatentation with all summaries of the same show
        Args:
            summaries_fp: Filepath to summaries JSON
            comments_fp: Filepath to comments JSON
            shows: If specified, returns comments only from those shows
            chunk_size: If > 0, splits summary into chunks of chunk_size words
        '''
        with open(summaries_fp) as f:
            summaries = json.load(f)
        self.summaries = {}
        for (show_name, ep_num, summary) in summaries:
            self.summaries[(show_name, ep_num)] = summary
        with open(comments_fp) as f:
            self.comments = json.load(f)
        if len(shows) > 0:
            self.comments = [c for c in self.comments if c[0] in shows]

        self.chunk_size = chunk_size
        if self.chunk_size > 0:
            self.summaries_chunks = {}
            for show_name, ep_num in self.summaries:
                self.summaries_chunks[(show_name, ep_num)] = []
                summary = self.summaries[(show_name, ep_num)]
                summary_as_sentences = nltk.sent_tokenize(summary) # this gives us a list of sentences

                sent_idx = 0
                while sent_idx < len(summary_as_sentences):
                    chunk = []
                    while True:
                        '''Keep adding sentences to chunk until the next sentence
                        would put it over the word limit or until we run out of sentences'''
                        if sent_idx >= len(summary_as_sentences):
                            break
                        tokenized_sentence = summary_as_sentences[sent_idx].split(' ')
                        if len(chunk) + len(tokenized_sentence) <= self.chunk_size:
                            chunk.extend(tokenized_sentence)
                            sent_idx += 1
                        else:
                            break
                    self.summaries_chunks[(show_name, ep_num)].append(' '.join(chunk))

                # summary_as_words = summary.split(' ')
                # chunks_as_words = [summary_as_words[i:i+self.chunk_size] for i in range(0, len(summary_as_words), self.chunk_size)]
                # chunks = [' '.join(chunk) for chunk in chunks_as_words]
                # self.summaries_chunks[(show_name, ep_num)] = chunks

class RelevantEpisodeTrainSet(BaseRelevantEpisodeSet):
    def __init__(self, summaries_fp, comments_fp, shows=set(), chunk_size=-1,
        negative_ratio=1, negative_sampling_strategy='random'):
        '''For each comment, returns its concatenation with the correct summary
        as well as with negative_ratio incorrect summaries
        Args:
            summaries_fp: Filepath to summaries JSON
            comments_fp: Filepath to comments JSON
            negative_ratio: If > 0, negative examples are packaged along batch dimension
            negative_sampling_strategy: Strategy used to select negative summaries for a comment
            chunk_size: If > 0, splits summary into chunks of chunk_size words
        '''
        super().__init__(summaries_fp, comments_fp, shows, chunk_size)

        self.negative_ratio = negative_ratio
        self.negative_sampling_strategy = negative_sampling_strategy

    def __len__(self):
        return len(self.comments) * (1 + self.negative_ratio)

    def sample_summary_chunk(self, show_name, ep_num):
        '''Samples a chunk from the summary corresponding to (show_name, ep_num)
        '''
        return random.choice(self.summaries_chunks[(show_name, ep_num)])

    def sample_negatives(self, positive_idx, num_to_sample=1):
        show_name, ep_num, query = self.comments[positive_idx]
        other_documents = [(other_show_name, other_ep_num) for (other_show_name, other_ep_num) in self.summaries.keys() 
            if other_show_name == show_name and other_ep_num != ep_num]

        if self.negative_sampling_strategy == 'random':
            negatives = random.sample(other_documents, num_to_sample)
        elif self.negative_sampling_strategy == 'bm25':
            if random.random() > 0.5:
                scores = get_bm25_scores([self.summaries[n] for n in other_documents], query)
                top_idx = np.argsort(scores)
                negatives = [other_documents[idx] for idx in top_idx[-num_to_sample:]]
            else:
                negatives = random.sample(other_documents, num_to_sample)
        else:
            raise ValueError()

        if self.chunk_size > 0:
            return [self.sample_summary_chunk(n[0], n[1]) for n in negatives]
        else:
            return [self.summaries[n] for n in negatives]

    def __getitem__(self, idx):
        positive_idx = idx // (1 + self.negative_ratio)
        show_name, ep_num, comment = self.comments[positive_idx]
        if idx % (1 + self.negative_ratio) == 0:
            if self.chunk_size > 0:
                summary = self.sample_summary_chunk(show_name, ep_num)
            else:
                summary = self.summaries[(show_name, ep_num)]
            target = [1]
        else:
            summary = self.sample_negatives(positive_idx, 1)[0]
            target = [0]

        return [summary], [comment], target

class RelevantEpisodeTestSet(BaseRelevantEpisodeSet):
    def __init__(self, summaries_fp, comments_fp, shows=set(), chunk_size=-1):
        '''For each comment, returns its concatentation with all summaries of the same show
        Args:
            summaries_fp: Filepath to summaries JSON
            comments_fp: Filepath to comments JSON
            shows: If specified, returns comments only from those shows
        '''
        super().__init__(summaries_fp, comments_fp, shows, chunk_size)

        # Pairings of a comment with all of the summaries from its show
        # (comment_idx, show_name, ep_num)
        self.pairings = []
        for comment_idx in range(len(self.comments)):
            for (show_name, ep_num) in self.summaries:
                if show_name == self.comments[comment_idx][0]:
                    if self.chunk_size > 0:
                        for summary_chunk_idx in range(len(self.summaries_chunks[(show_name, ep_num)])):
                            self.pairings.append((comment_idx, show_name, ep_num, summary_chunk_idx))
                    else:
                        self.pairings.append((comment_idx, show_name, ep_num))

        # Assign each summary an arbitrary, unique index (for passage aggregation later)
        self.summaries_idx = {}
        idx = 0
        for show_name, ep_num in self.summaries:
            self.summaries_idx[(show_name, ep_num)] = idx
            idx += 1

    def __len__(self):
        return len(self.pairings)

    def __getitem__(self, idx):
        if self.chunk_size > 0:
            comment_idx, show_name, ep_num, summary_chunk_idx = self.pairings[idx]
            summary = self.summaries_chunks[(show_name, ep_num)][summary_chunk_idx]
        else:
            comment_idx, show_name, ep_num = self.pairings[idx]
            summary = self.summaries[(show_name, ep_num)]

        _, true_ep_num, comment = self.comments[comment_idx]

        target = [1] if ep_num == true_ep_num else [0]

        return [summary], [comment], target, [comment_idx], [self.summaries_idx[(show_name, ep_num)]]

class RelevantEpisodeModule(pl.LightningDataModule):
    def __init__(self, summaries_fp, train_fp, val_fp, test_fp, model_name, 
        batch_size, negative_ratio=1, negative_sampling_strategy='random', 
        num_workers=8, comment_max_len=172, shows=set(), chunk_size=-1):
        '''Args:
            comment_max_len: Truncate comments to this length when using Longformer
            shows: If specified, returns comments only from those shows for Validation and Test sets
        '''
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name)
        if 'reformer' in self.hparams.model_name:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_config = AutoConfig.from_pretrained(self.hparams.model_name)
        self.trainset = RelevantEpisodeTrainSet(self.hparams.summaries_fp, self.hparams.train_fp, self.hparams.shows,
            self.hparams.chunk_size, self.hparams.negative_ratio, self.hparams.negative_sampling_strategy)
        self.valset = RelevantEpisodeTestSet(self.hparams.summaries_fp, self.hparams.val_fp, 
            self.hparams.shows, self.hparams.chunk_size)
        self.testset = RelevantEpisodeTestSet(self.hparams.summaries_fp, self.hparams.test_fp, 
            self.hparams.shows, self.hparams.chunk_size)
        
    def collate_fn_tokenize(self, batch):
        '''
        Args:
            batch: [[summaries_list, comments_list, targets_list, indexes_list], ]
        '''
        summaries = [summary for b in batch for summary in b[0]]
        comments = [comment for b in batch for comment in b[1]]
        targets = [target for b in batch for target in b[2]]

        if 'bigbird' in self.hparams.model_name:
            encoded_dict = self.tokenizer(
                    summaries,
                    comments,  
                    padding='longest',
                    return_tensors='pt',     # Return pytorch tensors.
                    truncation='only_second',
                    return_token_type_ids=True
            )
            num_global_tokens = 2 * self.model_config.block_size + \
                3 * self.model_config.block_size + self.model_config.num_random_blocks * self.model_config.block_size + \
                self.model_config.num_random_blocks * self.model_config.block_size
            if encoded_dict['input_ids'].shape[1] <= num_global_tokens:
                encoded_dict = self.tokenizer(
                    summaries,
                    comments,  
                    padding='max_length',
                    max_length=num_global_tokens + 1,
                    return_tensors = 'pt',     # Return pytorch tensors.
                    return_token_type_ids=True
                )
        elif 'longformer' in self.hparams.model_name:
            encoded_comments = self.tokenizer(comments, 
                max_length=self.hparams.comment_max_len, truncation=True)['input_ids']
            comments = self.tokenizer.batch_decode(encoded_comments, skip_special_tokens=True)
            encoded_dict = self.tokenizer(
                    summaries,
                    comments,  
                    padding='longest',
                    return_tensors='pt',     # Return pytorch tensors.
                    truncation='only_second',
                    return_token_type_ids=True
            )
            global_attention_mask = []
            for example in encoded_dict['input_ids']:
                sep_locations = ((example == self.model_config.sep_token_id).nonzero(as_tuple=True)[0])
                assert len(sep_locations) == 3
                global_tokens_mask = torch.zeros_like(example)
                global_tokens_mask[sep_locations[1]+1:sep_locations[2]] = 1
                global_attention_mask.append(global_tokens_mask)
            encoded_dict['global_attention_mask'] = torch.stack(global_attention_mask, dim=0)
        elif 'reformer' in self.hparams.model_name:
            encoded_dict = self.tokenizer(
                    summaries,
                    comments,  
                    padding='max_length',
                    max_length=np.prod(self.model_config.axial_pos_shape),
                    return_tensors = 'pt',     # Return pytorch tensors.
                    return_token_type_ids=True
            )
        elif 'nystromformer' in self.hparams.model_name:
            encoded_dict = self.tokenizer(
                    summaries,
                    comments,
                    max_length=self.model_config.max_position_embeddings,  
                    padding='longest',
                    return_tensors='pt',     # Return pytorch tensors.
                    truncation='only_second',
                    return_token_type_ids=True
            )
        elif 'roberta' in self.hparams.model_name:
            encoded_comments = self.tokenizer(comments, 
                max_length=self.hparams.comment_max_len, truncation=True)['input_ids']
            comments = self.tokenizer.batch_decode(encoded_comments, skip_special_tokens=True)
            encoded_dict = self.tokenizer(
                    summaries,
                    comments,  
                    padding='longest',
                    return_tensors='pt',     # Return pytorch tensors.
                    truncation='only_first',
                    return_token_type_ids=True
            )
        else:
            raise ValueError('Model not supported')

        if len(batch[0]) == 5:
            indexes = [index for b in batch for index in b[3]]
            summary_indexes = [index for b in batch for index in b[4]]
            return encoded_dict, torch.tensor(targets, dtype=bool), \
                torch.tensor(indexes, dtype=torch.long), torch.tensor(summary_indexes, dtype=torch.long)

        return encoded_dict, torch.tensor(targets, dtype=torch.long)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.trainset, 
                                           batch_size=self.hparams.batch_size, 
                                           shuffle=True,
                                           collate_fn=self.collate_fn_tokenize,
                                           num_workers=self.hparams.num_workers)
                                        
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valset, 
                                           batch_size=self.hparams.batch_size * 8, 
                                           shuffle=False,
                                           collate_fn=self.collate_fn_tokenize,
                                           num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.testset, 
                                           batch_size=self.hparams.batch_size * 8, 
                                           shuffle=False,
                                           collate_fn=self.collate_fn_tokenize,
                                           num_workers=self.hparams.num_workers)