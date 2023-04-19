import custom_sequence_classification as csc
from utils import prepare_data_for_modelling
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification, EarlyStoppingCallback, IntervalStrategy, set_seed
from datasets import load_dataset, ClassLabel
import torch
import hydra
from omegaconf import DictConfig
import numpy as np
import pandas as pd
import os
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


class ClassificationTrainer():
    def __init__(self, config):
        self.config = config
        self.seeds = self.config["seeds"]
        self.train_label = self.config["data"]["train_label"]
        self.test_label = self.config["data"]["test_label"]

        if self.config["data"]["emojis"]:
            self.emoji_tokens = open(self.config["data"]["emojis_names_path"], "r").read().replace("\n", " ").split(" ")
        else:
            self.emoji_tokens = None
        
        if self.config["data"]["meta_feats"]:
            self.author_feats = open(self.config["data"]["meta_feat_names_path"], "r").read().replace("\n", " ").split(" ")
        else:
            self.author_feats = None

        if not os.path.exists(self.config["logging"]["path"]):
            os.makedirs(self.config["logging"]["path"])


    def _prepare_data(self):
        data =  pd.read_csv(self.config["data"]["data_path"])
        if self.config["data"]["test_data_path"]:
            test_data = pd.read_csv(self.config["data"]["test_data_path"])

        for seed in self.seeds:
            # Split dataset (70/10/20)
            if self.config["data"]["test_data_path"]:
                _, data_test = train_test_split(test_data, test_size=0.2, random_state=seed, stratify=test_data[self.test_label])
                data_train, data_val = train_test_split(data, test_size=0.1, random_state=seed, stratify=data[self.test_label])
                data_test = data_test[list(data_train.columns)]
            else:
                data_train, data_test = train_test_split(data, test_size=0.2, random_state=seed, stratify=data[self.test_label])
                data_train, data_val = train_test_split(data_train, train_size=(0.7/0.8), test_size=(0.1/0.8), random_state=seed, stratify=data_train[self.test_label])

            # Different column names handling in case of e.g. training with weak labels
            if self.test_label != self.train_label:
                if self.test_label in data.columns:
                    data_train = data_train.drop(columns=[self.test_label])
                    data_val = data_val.drop(columns=[self.test_label])
                    data_test = data_test.drop(columns=[self.train_label])

                data_train = data_train.rename(columns={self.train_label: self.test_label})
                data_val = data_val.rename(columns={self.train_label: self.test_label})
                

            # Standardscaler
            if self.config["data"]["meta_feats"]:
                scaler = StandardScaler()
                data_train[self.author_feats[-18:]] = scaler.fit_transform(data_train[self.author_feats[-18:]])
                data_test[self.author_feats[-18:]] = scaler.transform(data_test[self.author_feats[-18:]])
                data_val[self.author_feats[-18:]] = scaler.transform(data_val[self.author_feats[-18:]])

            # Save Samples
            save_directory = "{}/seed_{}/".format(self.config["data"]["data_splitted_dir"], seed)
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            data_train.to_csv("{}data_train.csv".format(save_directory), index=False)
            data_val.to_csv("{}data_val.csv".format(save_directory), index=False)
            data_test.to_csv("{}data_test.csv".format(save_directory), index=False)


    def _read_prepared_data(self):
        data_dict = {}
        for seed in self.seeds:
            path_dir = "{}/seed_{}".format(self.config["data"]["data_splitted_dir"], seed)
            data_dict[seed] = load_dataset(path_dir, data_files={"train": "data_train.csv", "validation": "data_val.csv", "test": "data_test.csv"})
            label = ClassLabel(num_classes=self.config["num_classes"], names=[i for i in range(self.config["num_classes"])])
            data_dict[seed] = data_dict[seed].cast_column(self.test_label, label)
        self.data_dict = data_dict
        

    def train_and_test(self, model, dataset):
        train_args = TrainingArguments(
            output_dir="{}/trainer".format(self.config["logging"]["path"]),
            per_device_train_batch_size=self.config["train_args"]["batch_size_train"],
            per_device_eval_batch_size=self.config["train_args"]["batch_size_val"],
            evaluation_strategy=IntervalStrategy.STEPS,
            eval_steps=self.config["train_args"]["eval_steps"],
            save_steps=self.config["train_args"]["eval_steps"],
            save_total_limit=self.config["train_args"]["save_total_limit"],
            learning_rate=self.config["train_args"]["learning_rate"],
            num_train_epochs=self.config["train_args"]["num_train_epochs"],
            optim="adamw_hf",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            probs = torch.nn.functional.softmax(torch.from_numpy(logits), dim=-1)
            predictions = np.argmax(logits, axis=-1)
            accuracy = metrics.accuracy_score(y_true=labels, y_pred=predictions)
            recall = metrics.recall_score(y_true=labels, y_pred=predictions)
            precision = metrics.precision_score(y_true=labels, y_pred=predictions)
            f1 = metrics.f1_score(y_true=labels, y_pred=predictions)
            auc = metrics.roc_auc_score(labels, [i[1] for i in probs])
            aps = metrics.average_precision_score(labels, [i[1] for i in probs])
            return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "auc": auc, "aps": aps}

        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.config["train_args"]["early_stopping_patience"])]
        )

        # Fine tuning
        trainer.train()

        # Save model
        #trainer.save_model("{}".format(self.config["logging"]["path"]))

        # Test and return performance metrics
        return trainer.predict(dataset["test"])[2]


    def run_expirements(self):
        # Prepare data
        if not hasattr(self, "data_dict") and self.config["data"]["need_data_prep"]:
            self._prepare_data()
            self._read_prepared_data()
        elif not hasattr(self, "data_dict") and not self.config["data"]["need_data_prep"]:
            self._read_prepared_data()

        df_performance = pd.DataFrame(columns=["seed", "model", "train_data", "author_features"])
        for plm in self.config["plms"]:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(plm)
            if self.config["data"]["emojis"]:
                tokenizer.add_tokens(self.emoji_tokens)

            def tokenize_func(examples):
                return tokenizer(examples[self.config["data"]["text"]], padding="max_length", truncation=True, max_length=128)

            # Run expirements
            for seed in self.seeds:
                set_seed(seed)
                # Load model
                if self.config["data"]["meta_feats"]:
                    if plm == "bert-large-cased":
                        model = csc.CustomBertForSequenceClassification.from_pretrained(plm, num_labels=2, num_extra_dims=len(self.author_feats))
                    else:
                        model = csc.CustomRobertaForSequenceClassification.from_pretrained(plm, num_labels=2, num_extra_dims=len(self.author_feats))
                    data = prepare_data_for_modelling(
                        data=self.data_dict[seed],
                        tokenizer=tokenizer,
                        text_field=self.config["data"]["text"],
                        # additional_tokens=emoji_tokens, already added
                        padding="max_length",
                        truncation=True,
                        max_length=128,
                        extra_feats=self.author_feats,
                    )
                else:
                    model = AutoModelForSequenceClassification.from_pretrained(plm, num_labels=self.config["num_classes"])
                    data = self.data_dict[seed].map(tokenize_func, batched=True)

                if self.config["data"]["emojis"]:   
                    model.resize_token_embeddings(len(tokenizer))
                    
                if self.config["train_args"]["num_freeze_layers"]:
                    modules = [model.base_model.embeddings, model.base_model.encoder.layer[:self.config["train_args"]["num_freeze_layers"]]]
                    for module in modules:
                        for param in module.parameters():
                            param.requires_grad = False

                results = self.train_and_test(copy.deepcopy(model), data)
                results.update({"seed": seed, "model": plm, "train_data": self.config["data"]["train_data"], "author_features": self.config["data"]["meta_feats"]})
                df_performance = df_performance.append(results, ignore_index=True)
                df_performance.to_csv("{}/results_full_finetuning.csv".format(self.config["logging"]["path"], index=False))



@hydra.main(config_path="", config_name="config_hqp.yaml", version_base=None)
def main(cfg: DictConfig):
    ClassificationTrainer(cfg.conf_hqp).run_expirements()


if __name__ == '__main__':
    main()