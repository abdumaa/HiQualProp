# Script to combine text and numerical/categorical features for sequence classification
# Similar to: https://colab.research.google.com/drive/1F7COnwHqcLDPg_SS-oFgW3c2GPDWnS5Y#scrollTo=BAQFbN-wBpoz

import torch
from torch import nn
from transformers import RobertaForSequenceClassification, RobertaModel, BertForSequenceClassification, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput


class ClassificationHead(nn.Module):
    """Head for for sentence-level classification tasks."""

    def __init__(self, config, num_extra_dims):
        super().__init__()
        total_dims = config.hidden_size + num_extra_dims
        self.dense = nn.Linear(total_dims, total_dims)
        # self.dense_1 = nn.Linear(total_dims, config.hidden_size)
        # self.dense_2 = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob)
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(total_dims, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        # x = self.dense_1(x)
        x = torch.tanh(x)
        # x = self.dropout(x)
        # x = self.dense_2(x)
        # x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class CustomRobertaForSequenceClassification(RobertaForSequenceClassification):

    def __init__(self, config, num_extra_dims):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.roberta = RobertaModel(config)
        self.classifier = ClassificationHead(config, num_extra_dims)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        extra_data=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """Modified forward for sequence classification with additional features.

        Args:
            input_ids (Optional[torch.LongTensor], optional): Defaults to None.
            attention_mask (Optional[torch.FloatTensor], optional): Defaults to None.
            extra_data (Optional[torch.FloatTensor], optional): Defaults to None.
            token_type_ids (Optional[torch.LongTensor], optional): Defaults to None.
            position_ids (Optional[torch.LongTensor], optional): Defaults to None.
            head_mask (Optional[torch.FloatTensor], optional): Defaults to None.
            inputs_embeds (Optional[torch.FloatTensor], optional): Defaults to None.
            labels (Optional[torch.LongTensor], optional): Defaults to None.
            output_attentions (Optional[bool], optional): Defaults to None.
            output_hidden_states (Optional[bool], optional): Defaults to None.
            return_dict (Optional[bool], optional): Defaults to None.

        Returns:
            Union[Tuple, SequenceClassifierOutput]
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0] # dim: (batch_size, seq_length, hidden_size)
        cls_embedding = sequence_output[:, 0, :] # dim: (batch_size, hidden_size)
        output = torch.cat((cls_embedding, extra_data), dim=-1) # extra_data dim: (batch_size, num_extra_dims) -> output dim: (batch_size, hidden_size + num_extra_dims)

        logits = self.classifier(output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output


        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



class CustomBertForSequenceClassification(BertForSequenceClassification):

    def __init__(self, config, num_extra_dims):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)
        classifier_dropout = (config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob)
        self.dropout = nn.Dropout(classifier_dropout)
        total_dims = config.hidden_size + num_extra_dims
        self.classifier = nn.Linear(total_dims, config.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        extra_data=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """Modified forward for sequence classification with additional features.

        Args:
            input_ids (Optional[torch.LongTensor], optional): Defaults to None.
            attention_mask (Optional[torch.FloatTensor], optional): Defaults to None.
            extra_data (Optional[torch.FloatTensor], optional): Defaults to None.
            token_type_ids (Optional[torch.LongTensor], optional): Defaults to None.
            position_ids (Optional[torch.LongTensor], optional): Defaults to None.
            head_mask (Optional[torch.FloatTensor], optional): Defaults to None.
            inputs_embeds (Optional[torch.FloatTensor], optional): Defaults to None.
            labels (Optional[torch.LongTensor], optional): Defaults to None.
            output_attentions (Optional[bool], optional): Defaults to None.
            output_hidden_states (Optional[bool], optional): Defaults to None.
            return_dict (Optional[bool], optional): Defaults to None.

        Returns:
            Union[Tuple, SequenceClassifierOutput]
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        output = torch.cat((pooled_output, extra_data), dim=-1) # extra_data dim: (batch_size, num_extra_dims) -> output dim: (batch_size, hidden_size + num_extra_dims)
        output = self.dropout(output)
        logits = self.classifier(output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output


        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )