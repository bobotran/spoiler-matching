import comet_ml
from pytorch_lightning.loggers import CometLogger
import pytorch_lightning as pl
from dataset import RelevantEpisodeModule
from models.lightning_model import RelevanceRanker
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import shutil
import os
import argparse

def get_logger(exp_name):
    comet_logger = CometLogger(
            # api_key='', Add API key here for online mode
            save_dir='.',
            project_name='Spoiler Matching',
            experiment_name=exp_name
        )
    return comet_logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help="Filepath of config file")
    args = parser.parse_args()

    with open(args.config, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    out_dir = os.path.join(config['out_dir'], 'checkpoints', config['exp_name'])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    try:
        shutil.copy(args.config, out_dir)
    except shutil.SameFileError:
        pass

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
        chunk_size=340 if 'roberta' in config['model_name'] else -1
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    best_ckpt = ModelCheckpoint(
            dirpath=out_dir,
            filename='{val_mrr:.4f}',
            monitor='val_mrr',
            mode='max',
            save_top_k=2,
            save_last=True
        )

    logger = get_logger(config['exp_name'])
    logger.log_hyperparams(config)

    model = RelevanceRanker(config['model_name'], 
                            lr=config['lr'], 
                            lr_step_type=config['lr_step_type'], 
                            weight_decay=config['weight_decay'],
                            lr_interval=config['val_check_interval'] // config['device_batch_size'] if config['lr_step_type'] == 'step' else 1)

    accumulate = config['effective_batch_size'] // config['device_batch_size']
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[config['gpu']],
        accumulate_grad_batches=accumulate,
        logger=logger,
        log_every_n_steps=config['effective_batch_size'] // config['device_batch_size'],
        val_check_interval=config['val_check_interval'] // config['device_batch_size'] \
            if config['val_check_interval'] is not None else 1.0,
        callbacks=[lr_monitor, best_ckpt],
        max_epochs=100,
        precision=16
    )

    if len(config['resume_from']) == 0:
        trainer.fit(model, dm)
    else:
        trainer.fit(model, dm, ckpt_path=config['resume_from'])