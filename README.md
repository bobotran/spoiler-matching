# Spoiler Matching
This is the code for Spoiler Detection as Semantic Text Matching. The dataset along with a detailed description is available on [Kaggle](https://www.kaggle.com/datasets/bobotran/spoiler-matching) and [Hugging Face](https://huggingface.co/datasets/bobotran/spoiler-matching?clone=true).

# Quickstart
## Data
Start by downloading the dataset from [Kaggle](https://www.kaggle.com/datasets/bobotran/spoiler-matching) or [Hugging Face](https://huggingface.co/datasets/bobotran/spoiler-matching).

    mkdir data

and extract the dataset into `data/`.

## Environment
Please ensure that you have Anaconda or Miniconda installed, then

    conda env create -f environment.yml
    conda activate spoiler

## Logging
We use [Comet.ml](https://www.comet.com/docs/quick-start/) to store and read our logs. By default, `train.py` will run in offline mode, but you may enter your API key at the top of `train.py` to log your experiments on Comet.ml.

## Train

    python train.py --config config/longformer.yml

Pytorch Lightning model checkpoints are automatically saved in the checkpoints directory under the experiment name and top 2 models with best validation MRR are kept.

## Trained Models
Alternatively, you can skip training and [download the models from the paper](https://huggingface.co/bobotran/spoiler-matcher/tree/main?clone=true). 

## Test
Point the `resume_from` field in your config file (Ex: `checkpoints/longformer/longformer.yml`) to your desired model checkpoint, then

    python test.py --config checkpoints/longformer/longformer.yml --mode test

The individual MRR on the four test set shows will be printed first, then the total test set MRR.

# Auto-labeling
We provide a medium-size [autolabeled training set](https://www.kaggle.com/datasets/bobotran/spoiler-matching?select=matching) ready for training a spoiler matching model. But if you'd like to create your own training set, we also make available the raw unlabeled comments, as well as the irrelevant/relevant dataset we used to train the autolabeler.