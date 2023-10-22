import pytorch_lightning as pl
from dataset import RelevantEpisodeModule
from models.lightning_model import RelevanceRanker
import yaml
import shutil
import os
import argparse
import json
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help="Filepath of config file")
    parser.add_argument('--mode', required=True, choices=[ 'validate', 'test'])
    args = parser.parse_args()

    with open(args.config, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    model = RelevanceRanker.load_from_checkpoint(config['resume_from'])
    trainer = pl.Trainer(
            accelerator="gpu",
            devices=[config['gpu']],
            precision=16,
        )

    # Evaluate on shows individually first
    shows = set() 
    with open(config['val_fp'] if args.mode == 'validate' else config['test_fp']) as f:
        comments = json.load(f)
    for (show_name, _, _) in comments:
        shows.add(show_name)
    
    scores = []
    num_examples = []
    for show in shows:
        dm = RelevantEpisodeModule(
            config['summaries_fp'],
            config['train_fp'],
            config['val_fp'],
            config['test_fp'],
            config['model_name'],
            batch_size=config['device_batch_size'],
            negative_ratio=config['negative_ratio'],
            negative_sampling_strategy=config['negative_sampling_strategy'],
            num_workers=8,
            shows=set([show]),
            chunk_size=340 if 'roberta' in config['model_name'] else -1
        )

        if args.mode == 'validate':
            mrr = trainer.validate(model, dm)[0]['val_mrr']
            num_examples.append(len(dm.valset.comments))
        else:
            mrr = trainer.test(model, dm)[0]['test_mrr']
            num_examples.append(len(dm.testset.comments))
        scores.append(mrr)
        print('{} - {} MRR: {} over {} comments'.format(show, args.mode, mrr, num_examples[-1]))
    all_shows_score = sum([scores[idx] * num_examples[idx] for idx in range(len(scores))]) / sum(num_examples)
    print('All shows - {} MRR: {}'.format(args.mode, all_shows_score))