# Automatic prompt generation procedure as in LMBFF paper (https://arxiv.org/abs/2012.15723)
import logging
import time
import copy
import os
import hydra
from omegaconf import DictConfig
from openprompt.config import get_config_from_file
from openprompt.pipeline_base import PromptDataLoader, PromptForClassification
from openprompt.plms import load_plm
from openprompt.prompts import ManualVerbalizer, ManualTemplate
from openprompt.prompts.prompt_generator import LMBFFTemplateGenerationTemplate, T5TemplateGenerator, RobertaVerbalizerGenerator
from openprompt.trainer import ClassificationRunner
from openprompt.utils.reproduciblity import set_seed
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch
from transformers import AdamW
from tqdm import tqdm
from typing import OrderedDict
import yaml
from few_shot import utils
from few_shot.utils import classification_metrics


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomClassificationRunner(ClassificationRunner):

    def __init__(self,
                 model: PromptForClassification,
                 config = None,
                 train_dataloader = None,
                 valid_dataloader = None,
                 test_dataloader = None,
                 loss_function = None,
                 id2label = None,
                 k = None,
                 seed = None,
                 ):
        super().__init__(model = model,
                         config = config,
                         train_dataloader = train_dataloader,
                         valid_dataloader = valid_dataloader,
                         test_dataloader = test_dataloader,
                         loss_function = loss_function,
                         id2label = id2label
                        )
        self.k = k
        self.seed = seed

    def checkpoint_path(self, ckpt: str) -> str:
        return os.path.join(os.path.join(self.config.logging.path, "checkpoints"), f'k_{self.k}_{self.seed}_{ckpt}.ckpt')

    def test_epoch(self, split: str):
        outputs = []
        self.model.eval()
        with torch.no_grad():
            data_loader = self.valid_dataloader if split=='validation' else self.test_dataloader
            for batch_idx, batch in enumerate(tqdm(data_loader, desc=split)):
                batch = batch.to("cuda:{}".format(self.config.environment.local_rank)).to_dict()

                outputs.append( self.test_step(batch, batch_idx) )

        metrics = self.test_epoch_end(split, outputs)
        for metric_name, metric in metrics.items():
            self.log(f'{split}/{metric_name}', metric, self.cur_epoch)
        return metrics

    def test_step(self, batch, batch_idx):
        label = batch.pop('label')
        logits = self.model(batch)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        pred = torch.argmax(logits, dim=-1)
        return pred.cpu().tolist(), label.cpu().tolist(), probs.cpu().tolist()

    def test_epoch_end(self, split, outputs):
        preds = []
        labels = []
        probs = []
        for pred, label, prob_1 in outputs:
            preds.extend(pred)
            labels.extend(label)
            probs.extend(prob_1)

        self.save_results(split, {
            'preds': preds,
            'labels': labels,
            'probs': probs,
        })

        metrics = OrderedDict()
        for metric_name in self.config.classification.test_metric:
            metric = classification_metrics(preds, labels, probs, metric_name, id2label=self.id2label, label_path_sep=self.label_path_sep)
            metrics[metric_name] = metric
        return metrics

    def test(self, ckpt = None) -> dict:
        if ckpt:
            if not self.load_checkpoint(ckpt, load_state = False):
                exit()
        return self.test_epoch("test")

    def run(self, ckpt = None) -> dict:
        self.fit(ckpt)
        return self.test(ckpt = None if self.clean else 'best')

    def predict(self, split: str, ckpt = None) -> dict:
        if ckpt:
            if not self.load_checkpoint(ckpt, load_state = False):
                exit()

        outputs = []
        self.model.eval()
        with torch.no_grad():
            if split=='train':
                data_loader = self.train_dataloader
            elif split=='validation':
                data_loader = self.valid_dataloader
            else:
                data_loader = self.test_dataloader
            for batch_idx, batch in enumerate(tqdm(data_loader, desc=split)):
                batch = batch.to("cuda:{}".format(self.config.environment.local_rank)).to_dict()

                outputs.append( self.test_step(batch, batch_idx) )

        return outputs


class LMBFFPromptGenerator():

    def __init__(self, config):
        self.config = config
        self.seeds = self.config["seeds"]
        self.cuda = self.config["environment"]["cuda"]
        self.plm, self.tokenizer, self.model_config, self.WrapperClass = utils.custom_load_plm(
            model_name=self.config["plm"]["model_name"],
            model_path=self.config["plm"]["model_path"],
            num_freeze_layers=self.config["plm"]["num_freeze_layers"],
            )

        # Set label
        self.label = self.config["data"]["label"]
        
        # Load initial verbalizer (only used for prompt template generation!)
        with open(self.config["verbalizer_file_path"]) as f:
            lines = f.read().splitlines()

        self.verbalizer = ManualVerbalizer(
            tokenizer=self.tokenizer,
            num_classes=self.config["num_classes"],
            label_words=[[line] for line in lines]
        )

        # Set best template and best verbalizer to manual versions, will be replaced later if auto_v and/or auto_t are set to True
        self.best_template_dict = {}
        self.best_verbalizer_dict = {}
        path_prefix = "{}".format(self.config["logging"]["path"])
        for seed in self.seeds:
            if os.path.exists("{}best_templates/{}_best_template_k{}.txt".format(path_prefix, seed, self.config["k"])): 
                best_t = open("{}best_templates/{}_best_template_k{}.txt".format(path_prefix, seed, self.config["k"]), "r").read()
                self.best_template_dict[seed] = ManualTemplate(self.tokenizer, text=best_t)
            if os.path.exists("{}best_verbalizer/{}_best_verbalizer_k{}.txt".format(path_prefix, seed, self.config["k"])):
                best_v = open("{}best_verbalizer/{}_best_verbalizer_k{}.txt".format(path_prefix, seed, self.config["k"]), "r").read().replace("\n", " ").split(" ")
                self.best_verbalizer_dict[seed] = ManualVerbalizer(self.tokenizer, num_classes=self.config["num_classes"], label_words=best_v)


    def _prepare_data_fs(self):
        data =  pd.read_csv(self.config["data"]["dataset_file_path"])
        stratify_by = self.config["data"]["sample_hierarchy"][-1] # always stratify by last due to 1:n mapping
        for l in self.config["data"]["sample_hierarchy"]:
            if data[l].dtype == "O":
                data[f"label_{l}"] = pd.factorize(data[l])[0]
            else:
                data[f"label_{l}"] = data[l]

        for seed in self.seeds:
            # Split dataset (80/20)
            data_train, data_test = train_test_split(data, test_size=0.2, random_state=seed, stratify=data[stratify_by])
            data_train, data_val = train_test_split(data_train, train_size=(0.7/0.8), test_size=(0.1/0.8), random_state=seed, stratify=data_train[stratify_by])

            # Standardscaler
            if self.config["data"]["meta_feats"]:
                author_feats = open(self.config["data"]["author_feat_names_path"], "r").read().replace("\n", " ").split(" ")
                scaler = StandardScaler()
                data_train[author_feats] = scaler.fit_transform(data_train[author_feats])
                data_test[author_feats] = scaler.transform(data_test[author_feats])
                data_val[author_feats] = scaler.transform(data_val[author_feats])

            # Few Shot Sampler
            k = self.config["k"]
            k_dev = self.config["k"]
            if len(self.config["data"]["sample_hierarchy"]) == 2:
                dataset_train_0 = data_train[data_train["label_{}".format(self.config["data"]["sample_hierarchy"][0])]==0]
                dataset_train_1 = data_train[data_train["label_{}".format(self.config["data"]["sample_hierarchy"][0])]==1]
                _k = int(k/len(dataset_train_1["label_{}".format(self.config["data"]["sample_hierarchy"][-1])].value_counts()))
                fs_dataset_train_0 = dataset_train_0.sample(n=k, random_state=seed)
                fs_dataset_train_1 = dataset_train_1.groupby("label_{}".format(self.config["data"]["sample_hierarchy"][-1])).sample(n=_k, random_state=seed)
                fs_dataset_train = pd.concat([fs_dataset_train_0, fs_dataset_train_1], ignore_index=True)
                dataset_val_0 = data_val[data_val["label_{}".format(self.config["data"]["sample_hierarchy"][0])]==0]
                dataset_val_1 = data_val[data_val["label_{}".format(self.config["data"]["sample_hierarchy"][0])]==1]
                _k_dev = int(k_dev/len(dataset_val_1["label_{}".format(self.config["data"]["sample_hierarchy"][-1])].value_counts()))
                fs_dataset_val_0 = dataset_val_0.sample(n=k_dev, random_state=seed)
                fs_dataset_val_1 = dataset_val_1.groupby("label_{}".format(self.config["data"]["sample_hierarchy"][-1])).sample(n=_k_dev, random_state=seed, replace=True)
                fs_dataset_val = pd.concat([fs_dataset_val_0, fs_dataset_val_1], ignore_index=True)
                del dataset_train_0, dataset_train_1, fs_dataset_train_0, fs_dataset_train_1, dataset_val_0, dataset_val_1, fs_dataset_val_0, fs_dataset_val_1
            elif len(self.config["data"]["sample_hierarchy"]) == 1:
                fs_dataset_train = data_train.groupby(f"label_{self.label}").sample(n=k, random_state=seed)
                fs_dataset_val = data_val.groupby(f"label_{self.label}").sample(n=k_dev, random_state=seed)
            else:
                raise ValueError("sample_hierarchy in config must be of length either 1 or 2")

            # Save Samples
            save_directory = "{}/k_{}/seed_{}/".format(self.config["data"]["dataset_save_path"], self.config["k"], seed)
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            fs_dataset_train.to_csv("{}fs_dataset_train.csv".format(save_directory), index=False)
            fs_dataset_val.to_csv("{}fs_dataset_val.csv".format(save_directory), index=False)
            data_test.to_csv("{}data_test.csv".format(save_directory), index=False)


    def _read_prepared_data_fs(self):
        data_dict = {}
        for seed in self.seeds:
            path_prefix = "{}/k_{}/seed_{}/".format(self.config["data"]["dataset_save_path"], self.config["k"], seed)
            data_dict[seed] = {"pd": {}, "op": {}}
            data_dict[seed]["pd"]["train"] = pd.read_csv(f"{path_prefix}fs_dataset_train.csv")
            data_dict[seed]["pd"]["val"] = pd.read_csv(f"{path_prefix}fs_dataset_val.csv")
            data_dict[seed]["pd"]["test"] = pd.read_csv(f"{path_prefix}data_test.csv")
            data_dict[seed]["op"]["train"] = utils.transform_dataset_openprompt(data_dict[seed]["pd"]["train"], label_name=f"label_{self.label}", additional_text=self.label)
            data_dict[seed]["op"]["val"] = utils.transform_dataset_openprompt(data_dict[seed]["pd"]["val"], label_name=f"label_{self.label}", additional_text=self.label)
            data_dict[seed]["op"]["test"] = utils.transform_dataset_openprompt(data_dict[seed]["pd"]["test"], label_name=f"label_{self.label}", additional_text=self.label)
        
        self.data_dict = data_dict
        return data_dict
        

    def _fit(self, num_epochs, model, train_dataloader, val_dataloader, loss_func, optimizer, metric_for_best_model):
        best_score = 0.0
        if metric_for_best_model == "accuracy":
            idx = 0
        elif metric_for_best_model == "precision":
            idx = 1
        elif metric_for_best_model == "recall":
            idx = 2
        elif metric_for_best_model == "f1":
            idx = 3
        elif metric_for_best_model == "auc":
            idx = 4
        elif metric_for_best_model == "apr":
            idx = 5
        else:
            raise ValueError("metric_for_best_model should be one of the following: accuracy, precision, recall, f1, auc, apr")
        
        for epoch in range(num_epochs):
            self._train_epoch(model, train_dataloader, loss_func, optimizer)
            score = self._evaluate(model, val_dataloader)[idx]
            if score > best_score:
                best_score = score
        return best_score
    
    def _train_epoch(self, model, train_dataloader, loss_func, optimizer):
        #model.train()
        for step, inputs in enumerate(train_dataloader):
            if self.cuda:
                inputs = inputs.cuda()
            logits = model(inputs)
            labels = inputs["label"]
            loss = loss_func(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def _evaluate(self, model, val_dataloader):
        #model.eval()
        allpreds = []
        allprob = []
        alllabels = []
        with torch.no_grad():
            for step, inputs in enumerate(val_dataloader):
                if self.cuda:
                    inputs = inputs.cuda()
                logits = model(inputs)
                probs = torch.nn.functional.softmax(logits, dim=-1)
                labels = inputs["label"]
                alllabels.extend(labels.cpu().tolist())
                allprob.extend(probs[:,1].cpu().tolist())
                allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        # acc = sum([int(i==j) for i, j in zip(allpreds, alllabels)])/len(allpreds)
        acc = metrics.accuracy_score(alllabels, allpreds)
        prec = metrics.precision_score(alllabels, allpreds)
        rec = metrics.recall_score(alllabels, allpreds)
        f1 = metrics.f1_score(alllabels, allpreds)
        auc = metrics.roc_auc_score(alllabels, allprob)
        apr = metrics.average_precision_score(alllabels, allprob)
        return acc, prec, rec, f1, auc, apr, alllabels, allprob, allpreds


    def generate_template(self, train_dataset, val_dataset, template_candidates=None):
        if template_candidates is None:
            self.template_model, self.template_tokenizer, self.template_model_config, self.template_wrapper = load_plm(
                model_name=self.config["template_generator"]["plm"]["model_name"],
                model_path=self.config["template_generator"]["plm"]["model_path"]
            )
            # Load template structure
            with open(self.config["template_file_path"]) as f:
                template_txt = f.read()
            
            self.template = LMBFFTemplateGenerationTemplate(
                tokenizer=self.template_tokenizer,
                verbalizer=self.verbalizer,
                text=template_txt
            )
            if self.cuda:
                template_model = self.template_model.cuda()
            else:
                template_model = self.template_model
            template_generator = T5TemplateGenerator(
                model=template_model,
                tokenizer=self.template_tokenizer,
                tokenizer_wrapper=self.template_wrapper,
                verbalizer=self.verbalizer,
                beam_width=self.config["template_generator"]["beam_width"],
            )

            # DataLoader
            batch_size = len(train_dataset)
            dataloader = PromptDataLoader(
                dataset=train_dataset,
                template=self.template,
                tokenizer=self.template_tokenizer,
                tokenizer_wrapper_class=self.template_wrapper,
                batch_size=batch_size,
                decoder_max_length=self.config["template_generator"]["decoder_max_length"],
                max_seq_length=self.config["template_generator"]["max_seq_length"],
                shuffle=False,
                teacher_forcing=False,
            )
            for data in dataloader:
                if self.cuda:
                    data = data.cuda()
                template_generator._register_buffer(data)

            # Generate template candidates
            template_model.eval()
            with torch.no_grad():
                template_texts = template_generator._get_templates()
                original_template = self.template.text
                template_texts = [template_generator.convert_template(template_text, original_template) for template_text in template_texts]
            print("Template Candidates:", template_texts) # maybe save candidates?
            template_generator.release_memory()
            del template_generator, template_model
            torch.cuda.empty_cache()
        else:
            template_texts = template_candidates

        # Iterate over each candidate and select the best performing one on the validation set
        best_metrics = 0.0
        best_template_text = None
        for template_text in tqdm(template_texts):
            template = ManualTemplate(self.tokenizer, template_text)

            train_dataloader = PromptDataLoader(train_dataset, template, tokenizer=self.tokenizer, tokenizer_wrapper_class=self.WrapperClass, batch_size=self.config["template_generator"]["batch_size"])
            if len(val_dataset) > 64:
                val_batch_size = 64
            else:
                val_batch_size = len(val_dataset)
            val_dataloader = PromptDataLoader(val_dataset, template, tokenizer=self.tokenizer, tokenizer_wrapper_class=self.WrapperClass, batch_size=val_batch_size)

            model = PromptForClassification(copy.deepcopy(self.plm), template, self.verbalizer)

            runner = ClassificationRunner(model, self.config, train_dataloader, val_dataloader)
            runner.clean = True
            score = runner.fit()

            if score > best_metrics:
                print("New best score:", score)
                print("New best template:", template_text)
                best_metrics = score
                best_template_text = template_text

            del model, runner

        # Return best template candidate
        template = ManualTemplate(self.tokenizer, text=best_template_text)
        print(best_template_text)
        return template, best_template_text


    def generate_verbalizer(self, train_dataset, val_dataset, template, verbalizer_candidates=None):
        if verbalizer_candidates is None:
            if self.cuda:
                plm = self.plm.cuda()
            else:
                plm = self.plm
            verbalizer_generator = RobertaVerbalizerGenerator(
                model=plm,
                tokenizer=self.tokenizer,
                candidate_num=self.config["verbalizer_generator"]["candidate_num"],
                label_word_num_per_class=self.config["verbalizer_generator"]["label_word_num_per_class"]
            )

            if len(train_dataset) > 64:
                batch_size = 64
            else:
                batch_size = len(train_dataset)
            dataloader = PromptDataLoader(
                train_dataset,
                template,
                tokenizer=self.tokenizer,
                tokenizer_wrapper_class=self.WrapperClass,
                batch_size=batch_size
            )
            for data in dataloader:
                data = template.process_batch(data)
                if self.cuda:
                    data = data.cuda()
                verbalizer_generator.register_buffer(data)
            label_words_list = verbalizer_generator.generate()
            verbalizer_generator.release_memory()
            del verbalizer_generator, plm
            print(label_words_list)
        else:
            label_words_list = verbalizer_candidates

        # Iterate over each candidate and select the best one
        current_verbalizer = copy.deepcopy(self.verbalizer)
        best_metrics = 0.0
        best_label_words = None
        for label_words in tqdm(label_words_list):
            print(label_words)
            current_verbalizer.label_words = label_words
            train_dataloader = PromptDataLoader(train_dataset, template, tokenizer=self.tokenizer, tokenizer_wrapper_class=self.WrapperClass, batch_size=self.config["verbalizer_generator"]["batch_size"])
            if len(val_dataset) > 64:
                val_batch_size = 64
            else:
                val_batch_size = len(val_dataset)
            val_dataloader = PromptDataLoader(val_dataset, template, tokenizer=self.tokenizer, tokenizer_wrapper_class=self.WrapperClass, batch_size=val_batch_size)

            model = PromptForClassification(copy.deepcopy(self.plm), template, current_verbalizer)

            runner = ClassificationRunner(model, self.config, train_dataloader, val_dataloader)
            runner.clean = True
            score = runner.fit()

            if score > best_metrics:
                best_metrics = score
                best_label_words = label_words

            del model, runner

        # Return best verbalizer candidate
        verbalizer = ManualVerbalizer(self.tokenizer, num_classes=self.config["num_classes"], label_words=best_label_words)
        print(best_label_words)
        return verbalizer, best_label_words

    def evaluate(self, train_dataset, val_dataset, test_dataset, template, verbalizer, seed):
        # Main training loop
        train_dataloader = PromptDataLoader(train_dataset, template, tokenizer=self.tokenizer, tokenizer_wrapper_class=self.WrapperClass, batch_size=self.config["train"].batch_size)
        if len(val_dataset) > 64:
            val_batch_size = 64
        else:
            val_batch_size = len(val_dataset)
        val_dataloader = PromptDataLoader(val_dataset, template, tokenizer=self.tokenizer, tokenizer_wrapper_class=self.WrapperClass, batch_size=val_batch_size)
        test_dataloader = PromptDataLoader(test_dataset, template, tokenizer=self.tokenizer, tokenizer_wrapper_class=self.WrapperClass, batch_size=64)

        model = PromptForClassification(copy.deepcopy(self.plm), template, verbalizer)

        runner = CustomClassificationRunner(model, self.config, train_dataloader, val_dataloader, test_dataloader, k=self.config["k"], seed=seed)
        runner.clean = False

        return runner.run()

    def train(self, fs_train_dataset_op=None, fs_val_dataset_op=None, test_dataset_op=None):
        if fs_train_dataset_op is not None and fs_val_dataset_op is not None and test_dataset_op is not None:
            dataset_dict = {1: {"op": {"train": fs_train_dataset_op, "val": fs_val_dataset_op, "test": test_dataset_op}}}
        else:
            if self.config["data"]["need_data_prep"]:
                logger.info("Preparing data for label {}, k {}...".format(self.config["data"]["label"], self.config["k"]))
                self._prepare_data_fs()
            logger.info("Reading prepared data for label {}, k {}...".format(self.config["data"]["label"], self.config["k"]))
            dataset_dict = self._read_prepared_data_fs()           
        
        df_performances = pd.DataFrame({"PLM": self.config["plm"]["model_path"], "setting": "FS (k={})".format(self.config["k"]), "seed": self.seeds, "best_template": None, "best_verbalizer": None})
        for metric in self.config["classification"]["test_metric"]:
            df_performances[metric] = None

        for seed in self.seeds:
            logger.info("Starting run for label {}, k {}, seed {}...".format(self.config["data"]["label"], self.config["k"], seed))
            set_seed(seed)
            fs_train_dataset = dataset_dict[seed]["op"]["train"]
            fs_val_dataset = dataset_dict[seed]["op"]["val"]
            test_dataset = dataset_dict[seed]["op"]["test"]

            # Generate Template
            if self.config["auto_t"]:
                logger.info("Best Template Search...")
                start_time = time.time()
                self.best_template_dict[seed], best_template_text = self.generate_template(fs_train_dataset, fs_val_dataset)
                logger.info("Best Template Search finished, took {} seconds".format(time.time() - start_time))
                with open("{}best_templates/{}_best_template_k{}.txt".format(self.config["logging"]["path"], seed, self.config["k"]), "w") as text_file:
                    text_file.write(best_template_text)
                torch.cuda.empty_cache()
            else:
                if not hasattr(self, "best_template_dict"):
                    raise ValueError("If auto_t is set to false best templates have to be provided as {}best_templates/{}_best_template_k{}.txt".format(self.config["logging"]["path"], seed, self.config["k"]))
                print("Selecting Predefined Template")
            prompt_format = f"{self.best_template_dict[seed].wrap_one_example(fs_train_dataset[0])[0][1]['text']} {self.best_template_dict[seed].wrap_one_example(fs_train_dataset[0])[0][2]['text']} {self.best_template_dict[seed].wrap_one_example(fs_train_dataset[0])[0][3]['text']}"
            print("Selected Template:", prompt_format)
            df_performances.loc[df_performances["seed"]==seed, "best_template"] = prompt_format

            # Generate Verbalizer
            if self.config["auto_v"]:
                logger.info("Best Verbalizer Search...")
                start_time = time.time()
                self.best_verbalizer_dict[seed], best_label_words = self.generate_verbalizer(fs_train_dataset, fs_val_dataset, self.best_template_dict[seed])
                logger.info("Best Verbalizer Search finished, took {} seconds".format(time.time() - start_time))
                with open("{}best_verbalizer/{}_best_verbalizer_k{}.txt".format(self.config["logging"]["path"], seed, self.config["k"]), "w") as text_file:
                    text_file.write("\n".join(best_label_words))
                torch.cuda.empty_cache()
            else:
                if not hasattr(self, "best_verbalizer_dict"):
                    raise ValueError("If auto_v is set to false best verbalizer lists have to be provided as {}best_verbalizer/{}_best_verbalizer_k{}.txt".format(self.config["logging"]["path"], seed, self.config["k"]))
                print("Selecting Predefined Verbalizer")
            print("Selected Verbalizer:", self.best_verbalizer_dict[seed].label_words)

            df_performances.loc[df_performances["seed"]==seed, "best_verbalizer"] = " |".join([" &".join(i) for i in self.best_verbalizer_dict[seed].label_words])

            # Train and evaluate
            logger.info("Starting prompt-based finetuning...")
            start_time = time.time()
            metric_dict = self.evaluate(fs_train_dataset, fs_val_dataset, test_dataset, self.best_template_dict[seed], self.best_verbalizer_dict[seed], seed)
            logger.info("Prompt-based finetuning finished, took {} seconds".format(time.time() - start_time))
            for metric in self.config["classification"]["test_metric"]:
                df_performances.loc[df_performances["seed"]==seed, metric] = metric_dict[metric]

            df_performances.to_csv("{}/results_lmbff_RPROP_k{}.csv".format(self.config["logging"]["path"], self.config["k"]), index=False)
            torch.cuda.empty_cache()


    def forward(self, dataset, seed, split, template=None, verbalizer=None, model=None, ckpt=None):
        if template is None:
            if not hasattr(self, "best_template_dict"):
                raise ValueError("If template is not provided a best template has to be provided in {}best_templates/{}_best_template_k{}.txt".format(self.config["logging"]["path"], seed, self.config["k"]))
            template = self.best_template_dict[seed]

        if verbalizer is None:
            if not hasattr(self, "best_verbalizer_dict"):
                raise ValueError("If verbalizer is not provided a best verbalizer list has to be provided in {}best_verbalizer/{}_best_verbalizer_k{}.txt".format(self.config["logging"]["path"], seed, self.config["k"]))
            verbalizer = self.best_verbalizer_dict[seed]

        if model is None:
            model = PromptForClassification(copy.deepcopy(self.plm), template, verbalizer)

        if split=='train':
            if len(dataset) > 64:
                batch_size = 64
            else:
                batch_size = len(dataset)
            dataloader = PromptDataLoader(dataset, template, tokenizer=self.tokenizer, tokenizer_wrapper_class=self.WrapperClass, batch_size=batch_size)
            runner = CustomClassificationRunner(model, self.config, train_dataloader=dataloader, k=self.config["k"], seed=seed)
        elif split=='validation':
            if len(dataset) > 64:
                batch_size = 64
            else:
                batch_size = len(dataset)
            dataloader = PromptDataLoader(dataset, template, tokenizer=self.tokenizer, tokenizer_wrapper_class=self.WrapperClass, batch_size=batch_size)
            runner = CustomClassificationRunner(model, self.config, val_dataloader=dataloader, k=self.config["k"], seed=seed)
        else:
            dataloader = PromptDataLoader(dataset, template, tokenizer=self.tokenizer, tokenizer_wrapper_class=self.WrapperClass, batch_size=64)
            runner = CustomClassificationRunner(model, self.config, test_dataloader=dataloader, k=self.config["k"], seed=seed)

        runner.clean = False

        return runner.predict(split=split, ckpt=ckpt)



@hydra.main(config_path="", config_name="config_lmbff.yaml", version_base=None)
def main(cfg: DictConfig):
    LMBFFPromptGenerator(cfg.conf_lmbff).train()


if __name__ == '__main__':
    main()