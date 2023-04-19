# from https://github.com/VinAIResearch/BERTweet/blob/master/TweetNormalizer.py

from emoji import demojize
from nltk.tokenize import TweetTokenizer
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score


tokenizer = TweetTokenizer()


def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token
        

def normalizeTweet(tweet):
    tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "...").replace("🏻\u200d♂", ""))
    normTweet = " ".join([normalizeToken(token) for token in tokens])

    normTweet = (
        normTweet.replace("cannot ", "can not ")
        .replace("n't ", " n't ")
        .replace("n 't ", " n't ")
        .replace("ca n't", "can't")
        .replace("ai n't", "ain't")
    )
    normTweet = (
        normTweet.replace("'m ", " 'm ")
        .replace("'re ", " 're ")
        .replace("'s ", " 's ")
        .replace("'ll ", " 'll ")
        .replace("'d ", " 'd ")
        .replace("'ve ", " 've ")
    )
    normTweet = (
        normTweet.replace(" p . m .", "  p.m.")
        .replace(" p . m ", " p.m ")
        .replace(" a . m .", " a.m.")
        .replace(" a . m ", " a.m ")
    )
    normTweet = (
        normTweet.replace("🇷 🇺", ":Russia:")
        .replace("🇺 🇦", ":Ukraine:")
        .replace("🇺 🇸", ":United_States:")
        .replace("🇺 🇳", ":United_Nations:")
        .replace("🇬 🇧", ":United_Kingdom:")
        .replace("🇹 🇷", ":Turkey:")
        .replace("🇵 🇰", ":Pakistan:")
    )


    return " ".join(normTweet.split())



def prepare_data_for_modelling(data, tokenizer, text_field, additional_tokens=None, padding="max_length", truncation=True, max_length=128, extra_feats=None):
    # Tokenization
    if additional_tokens:
        tokenizer.add_tokens(additional_tokens)

    def tokenize_func(examples):
        return tokenizer(examples[text_field], padding=padding, truncation=truncation, max_length=max_length)

    data = data.map(tokenize_func, batched=True)

    # Handle extra features
    if extra_feats:
        for key in data.keys():
            df = data[key]
            extra_data = np.ndarray(shape=(df.num_rows,0))
            for extra_feat in extra_feats:
                extra_data = np.c_[extra_data, df[extra_feat]]

            df = Dataset.from_dict(
                {
                    text_field: df[text_field],
                    "extra_data": extra_data,
                    "labels": df["labels"],
                    "input_ids": df["input_ids"],
                    "attention_mask": df["attention_mask"]
                }
            )

            data[key] = df

    return data


def transform_dataset_openprompt(dataset, label_name, additional_text="", extra_feats=None):

    def transform(example):
        text_a = example["text_normalized"]
        if additional_text:
            text_b = example[additional_text]
        else:
            text_b = ""
        if extra_feats:
            meta = {extra_feat: example[extra_feat] for extra_feat in extra_feats}
        else:
            meta = None
        label = int(example[label_name])
        guid = "{}".format(example["id"])
        return InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, meta=meta)

    if isinstance(dataset, DatasetDict):
        openprompt_df = {}
        for key in dataset.keys():
            openprompt_df[key] = list(map(transform, dataset[key]))
    elif isinstance(dataset, Dataset):
        openprompt_df = list(map(transform, dataset))
    elif isinstance(dataset, pd.DataFrame):
        openprompt_df = list(dataset.apply(lambda x: transform(x), axis=1))
    else:
        raise ValueError("Not recognized type of dataset")

    return openprompt_df


def classification_metrics(preds, labels, prob_1, metric="micro-f1", id2label=None, label_path_sep='-'):
    """From https://github.com/thunlp/OpenPrompt/blob/17283c194bf76fd4c06fa89bdf7f947026e53e68/openprompt/utils/metrics.py#L57 but added auc and apr
    evaluation metrics for classification task.
    Args:
        preds (Sequence[int]): predicted label ids for each examples
        labels (Sequence[int]): gold label ids for each examples
        metric (str, optional): type of evaluation function, support 'micro-f1', 'macro-f1', 'accuracy', 'precision', 'recall'. Defaults to "micro-f1".
    Returns:
        score (float): evaluation score
    """
    
    if metric == "micro-f1":
        score = f1_score(labels, preds, average="micro")
    elif metric == "macro-f1":
        score = f1_score(labels, preds, average="macro")
    elif metric == "weighted-f1":
        score = f1_score(labels, preds, average="weighted")
    elif metric == "f1":
        score = f1_score(labels, preds)
    elif metric == "accuracy":
        score = accuracy_score(labels, preds)
    elif metric == "precision":
        score = precision_score(labels, preds)
    elif metric == "weighted-precision":
        score = precision_score(labels, preds, average="weighted")
    elif metric == "recall":
        score = recall_score(labels, preds)
    elif metric == "weighted-recall":
        score = recall_score(labels, preds, average="weighted")
    elif metric == "auc":
        score = roc_auc_score(labels, [i[1] for i in prob_1]) #temp
    elif metric == "weighted-auc":
        score = roc_auc_score(labels, prob_1, average="weighted", multi_class="ovr")
    elif metric == "aps":
        score = average_precision_score(labels, [i[1] for i in prob_1], average="weighted") #temp
    else:
        raise ValueError("'{}' is not a valid evaluation type".format(metric))
    return score


def custom_load_plm(model_name, model_path, specials_to_add=None, num_freeze_layers=None):
    r"""Load a pretrained model and freeze some layers (from openprompt but slightly modified)
    Args:
        model_name (str): name of the pretrained model
        model_path (str): path to the pretrained model
        num_freeze_layers (int, optional): number of layers to freeze. Defaults to None.
    Returns:
        :obj:`PreTrainedModel`: The pretrained model.
        :obj:`tokenizer`: The pretrained tokenizer.
        :obj:`model_config`: The config of the pretrained model.
        :obj:`wrapper`: The wrapper class of this plm.
    """

    model, tokenizer, model_config, wrapper = load_plm(model_name, model_path, specials_to_add=specials_to_add)

    if num_freeze_layers:
        modules = [model.base_model.embeddings, *model.base_model.encoder.layer[:num_freeze_layers]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
                
    return model, tokenizer, model_config, wrapper