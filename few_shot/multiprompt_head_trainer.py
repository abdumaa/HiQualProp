from openprompt.config import get_config_from_file
from few_shot import lmbff_trainer as lmbff
from few_shot.utils_nn import NeuralNetworkTrainer
import xgboost as xgb
import pandas as pd
import os
import logging
import time
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score
from sklearn.model_selection import ParameterGrid


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiPromptHead():
    def __init__(self, config_mp):
        # Load configs
        ## Multi prompt head
        self.config_mp = config_mp

        ## Binary class prompting
        self.config_lmbff_bc = get_config_from_file(self.config_mp["paths"]["config_lmbff_bc"])

        ## Multi class prompting
        self.config_lmbff_mc = get_config_from_file(self.config_mp["paths"]["config_lmbff_mc"])

        ## Overwrite configs if specified
        if self.config_mp["prompt_models"]["template_train_batch_size"] is not None:
            self.config_lmbff_bc["template_generator"]["batch_size"] = self.config_mp["prompt_models"]["template_train_batch_size"]
            self.config_lmbff_mc["template_generator"]["batch_size"] = self.config_mp["prompt_models"]["template_train_batch_size"]

        if self.config_mp["prompt_models"]["verbalizer_train_batch_size"] is not None:
            self.config_lmbff_bc["verbalizer_generator"]["batch_size"] = self.config_mp["prompt_models"]["verbalizer_train_batch_size"]
            self.config_lmbff_mc["verbalizer_generator"]["batch_size"] = self.config_mp["prompt_models"]["verbalizer_train_batch_size"]
            
        if self.config_mp["prompt_models"]["train_batch_size"] is not None:
            self.config_lmbff_bc["train"]["batch_size"] = self.config_mp["prompt_models"]["train_batch_size"]
            self.config_lmbff_mc["train"]["batch_size"] = self.config_mp["prompt_models"]["train_batch_size"]

        if self.config_mp["k"] is not None:
            self.config_lmbff_bc["k"] = self.config_mp["k"]
            self.config_lmbff_mc["k"] = self.config_mp["k"]

        if self.config_mp["seeds"] is not None:
            self.config_lmbff_bc["seeds"] = self.config_mp["seeds"]
            self.config_lmbff_mc["seeds"] = self.config_mp["seeds"]

        # Initialize prompt models
        self.pg_bc = lmbff.LMBFFPromptGenerator(self.config_lmbff_bc) 
        self.pg_mc = lmbff.LMBFFPromptGenerator(self.config_lmbff_mc)

        # Initialize data dicts and feature sets
        for seed in self.config_mp["seeds"]:
            save_directory = "{}/k_{}/seed_{}/".format(self.config_lmbff_bc["data"]["dataset_save_path"], self.config_lmbff_bc["k"], seed)
            if not os.path.exists(save_directory):
                self.pg_bc._prepare_data_fs()
        self.data_dict_bc = self.pg_bc._read_prepared_data_fs()
        self.data_dict_mc = self.pg_mc._read_prepared_data_fs()
        self.meta_feats = open(self.config_mp["paths"]["meta_feat_names_path"], "r").read().replace("\n", " ").split(" ")
        self.truncated_meta_feats = open(self.config_mp["paths"]["truncated_meta_feat_names_path"], "r").read().replace("\n", " ").split(" ")
        self.author_feats = open(self.config_mp["paths"]["author_feat_names_path"], "r").read().replace("\n", " ").split(" ")
        self.account_feats = open(self.config_mp["paths"]["account_feat_names_path"], "r").read().replace("\n", " ").split(" ")

        if not os.path.exists(self.config_mp["paths"]["performance_logging"]):
            os.makedirs(self.config_mp["paths"]["performance_logging"])

    
    def get_prompt_forwards_probs(self):
        """Trains prompt models if needed and stores forwards probs in self.data_dict_bc."""
        if self.config_mp["prompt_models"]["need_train_bc"]:
            self.pg_bc.train()
        if self.config_mp["prompt_models"]["need_train_mc"]:
            self.pg_mc.train()

        for seed in self.data_dict_bc.keys():
            for set in self.data_dict_bc[seed]["op"].keys():
                output_bc = self.pg_bc.forward(dataset=self.data_dict_bc[seed]["op"][set], seed=seed, split="test", ckpt="best")
                output_mc = self.pg_mc.forward(dataset=self.data_dict_bc[seed]["op"][set], seed=seed, split="test", ckpt="best")
                probs_bc = []
                for _, _, prob in output_bc:
                    probs_bc.extend([p[1:] for p in prob])
                probs_mc = []
                for _, _, prob in output_mc:
                    probs_mc.extend([p[1:] for p in prob])

                prob_bc_col_names = [f"prob_{i+1}" for i, v in enumerate(probs_bc[0])]
                prob_mc_col_names = [f"prob_mc_{i+1}" for i, v in enumerate(probs_mc[0])]

                self.data_dict_bc[seed]["pd"][set][prob_bc_col_names] = probs_bc
                self.data_dict_bc[seed]["pd"][set][prob_mc_col_names] = probs_mc

        prob_bc_col_names.extend(prob_mc_col_names)
        self.prob_feats = prob_bc_col_names
        for prob in prob_bc_col_names:
            if prob not in self.meta_feats:
                self.meta_feats.append(prob)
            if prob not in self.truncated_meta_feats:
                self.truncated_meta_feats.append(prob)
            if prob not in self.author_feats:
                self.author_feats.append(prob)
            if prob not in self.account_feats:
                self.account_feats.append(prob)


    def elastic_net_head_train_val(self, train_data, features, val_data, model, params):
        model.set_params(**params)
        model.fit(train_data[features], train_data["label_labels"])
        preds_val = model.predict(val_data[features])
        probs_val = model.predict_proba(val_data[features])
        performance_dict = {
            "accuracy": accuracy_score(val_data["label_labels"], preds_val),
            "precision": precision_score(val_data["label_labels"], preds_val),
            "recall": recall_score(val_data["label_labels"], preds_val),
            "f1": f1_score(val_data["label_labels"], preds_val),
            "auc": roc_auc_score(val_data["label_labels"], probs_val[:, 1]),
            "aps": average_precision_score(val_data["label_labels"], probs_val[:, 1]),
        }
        return performance_dict
    

    def nn_head_train_val(self, train_data, features, val_data, test_data, params):
        model = NeuralNetworkTrainer(params, len(features))
        model.train(train_data, val_data, "label_labels", features)
        return model.inference_loop(test_data, "label_labels", features, ["accuracy", "precision", "recall", "f1", "auc", "aps"])

    
    def xgboost_head_train_val(self, train_data, features, val_data, test_data, model, params):
        model.set_params(**params)
        model.fit(train_data[features], train_data["label_labels"], eval_set=[(val_data[features], val_data["label_labels"])])
        preds_val = model.predict(test_data[features])
        probs_val = model.predict_proba(test_data[features])
        performance_dict = {
            "accuracy": accuracy_score(test_data["label_labels"], preds_val),
            "precision": precision_score(test_data["label_labels"], preds_val),
            "recall": recall_score(test_data["label_labels"], preds_val),
            "f1": f1_score(test_data["label_labels"], preds_val),
            "auc": roc_auc_score(test_data["label_labels"], probs_val[:, 1]),
            "aps": average_precision_score(test_data["label_labels"], probs_val[:, 1]),
        }
        return performance_dict


    def hp_tuning(self, train_data, features, val_data, model_type, seed):
        if type(self.config_mp["models"][model_type]) == DictConfig:
            paramgrid = OmegaConf.to_container(self.config_mp["models"][model_type])
        else:
            paramgrid = self.config_mp["models"][model_type]
        if model_type == "elasticnet":
            model = LogisticRegression(penalty="elasticnet", random_state=seed, solver="saga", max_iter=10000, n_jobs=-1)
            best_score = 0.0
            for g in ParameterGrid(paramgrid):
                score = self.elastic_net_head_train_val(train_data, features, val_data, model, g)["f1"]
                if score > best_score:
                    best_score = score
                    best_grid = g

        elif model_type == "neuralnet":
            best_score = 0.0
            for g in ParameterGrid(paramgrid):
                score = self.nn_head_train_val(train_data, features, val_data, val_data, g)["f1"]
                if score > best_score:
                    best_score = score
                    best_grid = g

        elif model_type == "xgboost":
            model = xgb.XGBClassifier()
            best_score = 0.0
            for g in ParameterGrid(paramgrid):
                score = self.xgboost_head_train_val(train_data, features, val_data, val_data, model, g)["f1"]
                if score > best_score:
                    best_score = score
                    best_grid = g

        else:
            raise ValueError(f"model_type {model_type} not defined")

        return best_score, best_grid


    def model_test(self, train_data, features, val_data, test_data, model_type, seed, params):
        if model_type == "elasticnet":
            model = LogisticRegression(penalty="elasticnet", random_state=seed, solver="saga", max_iter=10000, n_jobs=-1)
            performance_dict = self.elastic_net_head_train_val(train_data, features, test_data, model, params)

        elif model_type == "neuralnet":
            performance_dict = self.nn_head_train_val(train_data, features, val_data, test_data, params)

        elif model_type == "xgboost":
            model = xgb.XGBClassifier()
            performance_dict = self.xgboost_head_train_val(train_data, features, val_data, test_data, model, params)

        else:
            raise ValueError(f"model_type {model_type} not defined")

        return performance_dict


    def run_experiments(self):
        logger.info("Preparing data for k {}...".format(self.config["k"]))
        self.get_prompt_forwards_probs()
        df_performance = pd.DataFrame(columns=["seed", "model", "feature_set", "best_grid", "accuracy", "precision", "recall", "f1", "auc", "aps"])
        for seed in self.data_dict_bc.keys():
            for model_type in list(self.config_mp["models"].keys()):
                for feature_set in self.config_mp["features"]["feature_sets"]:
                    logger.info("Starting run for k {}, model {}, feature set {}, seed {}...".format(self.config["k"], model_type, feature_set, seed))
                    if feature_set == "meta_feats":
                        features = self.meta_feats
                    elif feature_set == "truncated_meta_feats":
                        features = self.truncated_meta_feats
                    elif feature_set == "author_feats":
                        features = self.author_feats
                    elif feature_set == "account_feats":
                        features = self.account_feats
                    else:
                        features = self.prob_feats
                    if self.config_mp["cross_fitting"]:
                        logger.info("Starting HP-Tuning...")
                        start_time = time.time()
                        best_score, best_grid = self.hp_tuning(self.data_dict_bc[seed]["pd"]["val"], features, self.data_dict_bc[seed]["pd"]["train"], model_type, seed)
                        logger.info("HP-Tuning finished, took {} seconds".format(time.time() - start_time))
                        logger.info("Starting testing...")
                        start_time = time.time()
                        results = self.model_test(self.data_dict_bc[seed]["pd"]["val"], features, self.data_dict_bc[seed]["pd"]["train"], self.data_dict_bc[seed]["pd"]["test"], model_type, seed, best_grid)
                        logger.info("Testing finished, took {} seconds".format(time.time() - start_time))
                    else:
                        best_score, best_grid = self.hp_tuning(self.data_dict_bc[seed]["pd"]["train"], features, self.data_dict_bc[seed]["pd"]["val"], model_type, seed)
                        results = self.model_test(self.data_dict_bc[seed]["pd"]["train"], features, self.data_dict_bc[seed]["pd"]["val"], self.data_dict_bc[seed]["pd"]["test"], model_type, seed, best_grid)
                    results.update({"seed": seed, "model": model_type, "feature_set": feature_set, "best_grid": best_grid})
                    df_performance = df_performance.append(results, ignore_index=True)
                    df_performance.to_csv("{}results_multiprompt_RPROP_k{}.csv".format(self.config_mp["paths"]["performance_logging"], self.config_mp["k"]), index=False)



@hydra.main(config_path="", config_name="config_multiprompt.yaml", version_base=None)
def main(cfg: DictConfig):
    MultiPromptHead(cfg.conf_multiprompt).run_experiments()


if __name__ == '__main__':
    main()