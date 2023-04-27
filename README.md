# HQP

This repository contains the data and the implementation of the experiments of the paper *HQP: A Human-Annotated Dataset for Detecting Online Propaganda*.


## Overview

In this work we present HQP, a high-quality human-annotated dataset for detecting online propaganda. Our work additionally includes:

1. Experiments on the performance of fully fine-tuning state-of-the-art pretrained language models on the task of detecting online propaganda on our dataset.
2. Experiments on the performance of state-of-the-art few-shot learning (prompt-based learning with [LMBFF](https://arxiv.org/pdf/2012.15723.pdf)) on our dataset.
3. An extension of prompt-based learning to the setting of multiple connected labels and to handle numerical features.
4. Experiments on the performance of incorporating author features in both full fine-tuning and few-shot learning.

You can find more details of this work in our paper.


## Requirements

To run our code, please install all the dependency packages by using the following command:

```
pip install -r requirements.txt
```

## Data

**We only share tweet-ids and labels and not processed datasets in this repository due to Twitter privacy policy. However, upon individual requests (`blinded`), we can make the data available.**

We keep our annotated HQP dataset in `Data/HiQualProp`.

For few-shot learning, data samples would be generated to `few-shot/data_splitted` automatically when experiments are executed.

## Full fine-tuning

To replicate full fine-tuning as in our work, execute the following command:

```bash
python full/classification_trainer.py --multirun conf_hqp=hiqualprop,hiqualprop_mf,twe,tweetspin,weaklabels
```

This will fine-tune [BERT-large](https://aclanthology.org/N19-1423.pdf), [RoBERTa-large](https://arxiv.org/pdf/1907.11692.pdf), and [BERTweet-large](https://aclanthology.org/2020.emnlp-demos.2.pdf) (each 5 runs by default) on:
* our HQP dataset (hiqualprop)
* our HQP dataset while incorporating author features (hiqualprop_mf)
* the [TWE](https://truthandtrustonline.com/wp-content/uploads/2020/10/TTO03.pdf) dataset (twe)
* the replicated [TWEETSPIN](https://aclanthology.org/2022.naacl-main.251.pdf) dataset (tweetspin)
* weak labels from our dataset (weaklabels)

Alternatively they can be executed seperately using e.g.:

```bash
python full/classification_trainer.py conf_hqp=hiqualprop
```

Performance evaluation and logging will be generated to the designated folder in `full`. The config files are in `full/conf_hqp`.


## Few-shot learning

To replicate prompt-based learning as in our work, execute the following command:

```bash
python few_shot/multiprompt_head_trainer.py --multirun conf_multiprompt=k16,k32,k64,k128
```

This will first perform (5 runs each) prompt-based learning as in [LMBFF](https://arxiv.org/pdf/2012.15723.pdf) for the different sample sizes (16, 32, 64, 128) and for the two labels (BL and PSL). Then the different classification heads (elastic net and neural net) are trained on the varbalizer probabilites (either alone, or including different sets of author features) for the different sample sizes (again 5 runs each). Evaluation and logging for both LMBFF and the classification heads are generated to the designated folders in `few_shot`. The configs are in `few_shot/conf_multiprompt`.

If you wish to only execute LMBFF training then execute the following command:

```bash
python few_shot/lmbff_trainer.py --multirun conf_lmbff=k16_bc,k32_bc,k64_bc,k128_bc,k16_mc,k32_mc,k64_mc,k128_mc
```

Where _bc performs LMBFF for the binary propaganda label and _mc for the propaganda strategy label. The configs are in `few_shot/conf_lmbff`.

Again, both procedures can be executed for different sample sizes seperately, e.g.:

```bash
python few_shot/lmbff_trainer.py conf_lmbff=k16_bc
```


## Bugs or questions?

Please address any issues rergarding the code to \[Anonymous\] (`anonymous`).