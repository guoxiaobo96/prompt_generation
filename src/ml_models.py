import torch
import torch.nn as nn
from transformers.models.bert import BertPreTrainedModel, BertForSequenceClassification, BertModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.models.roberta import RobertaForSequenceClassification, RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaLMHead, RobertaClassificationHead
from transformers.modeling_outputs import SequenceClassifierOutput

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
import itertools
import multiprocessing


def resize_token_type_embeddings(model, new_num_types: int, random_segment: bool):
    """
    Resize the segment (token type) embeddings for BERT
    """
    if hasattr(model, 'bert'):
        old_token_type_embeddings = model.bert.embeddings.token_type_embeddings
    else:
        raise NotImplementedError
    new_token_type_embeddings = nn.Embedding(new_num_types, old_token_type_embeddings.weight.size(1))
    if not random_segment:
        new_token_type_embeddings.weight.data[:old_token_type_embeddings.weight.size(0)] = old_token_type_embeddings.weight.data

    model.config.type_vocab_size = new_num_types
    if hasattr(model, 'bert'):
        model.bert.embeddings.token_type_embeddings = new_token_type_embeddings
    else:
        raise NotImplementedError



from sklearn.metrics import precision_recall_fscore_support

def filter_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def res_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def acc_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
        'acc': acc,
    }



def init(train_logits, train_labels):
    global logits, labels
    logits = train_logits
    labels = train_labels

def eval_label(pairing
):
    global logits, labels
    label_logits = np.take(logits, pairing, axis=-1)
    preds = np.argmax(label_logits, axis=-1)
    correct = np.sum(preds == labels)
    return correct / len(labels)



class BertForPromptFinetuning(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For label search.
        self.return_full_softmax = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        mask_pos=None,
        labels=None,
    ):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.cls(sequence_mask_output)

        # Exit early and only return mask logits.
        if self.return_full_softmax:
            if labels is not None:
                return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits) # Log prob of right polarity

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
        return ((loss,) + output) if loss is not None else output



class RobertaForPromptFinetuning(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        self.lm_head = RobertaLMHead(config)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For auto label search.
        self.return_full_softmax = None

    def get_label(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
    ):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask
        )

        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.lm_head(sequence_mask_output)
        logits = prediction_mask_scores.detach().cpu().numpy()
        labels = labels.cpu().numpy()

        indices = list()
        num_labels = np.max(labels) + 1
        # indices = [self.tokenizer.vocab["Ġterrible"],self.tokenizer.vocab["Ġgreat"]]
        for idx in range(num_labels):
            label_logits = logits[labels == idx]
            scores = label_logits.mean(axis=0)
            kept = []
            for i in np.argsort(-scores):
                text = self.vocab[i]
                if not text.startswith("Ġ"):
                    continue
                kept.append(i)
            # indices.extend(kept[:100])
            indices.append(kept[:100])
        
        valid_indices = [sorted(list(set(inds))) for inds in indices]
        pairings = list(itertools.product(*valid_indices))

        pairing_scores = []
        # with multiprocessing.Pool(initializer=init, initargs=(logits, labels)) as workers:
        #     chunksize = max(10, int(len(pairings) / 1000))
        #     for score in workers.imap(eval_label, pairings, chunksize=chunksize):
        #         pairing_scores.append(score)
        for pairing in pairings:
            label_logits = np.take(logits, pairing, axis=-1)
            preds = np.argmax(label_logits, axis=-1)
            correct = np.sum(preds == labels)
            pairing_scores.append(correct / labels.size)

        sorted_score = np.sort(-np.array(pairing_scores), kind='mergesort')
        sorted_indices = np.argsort(-np.array(pairing_scores), kind='mergesort')
        best_idx = np.argsort(-np.array(pairing_scores), kind='mergesort')[:1]
        best_score = [pairing_scores[i] for i in best_idx]
        indices = [pairings[i] for i in best_idx]
        return indices[0]


    def forward(
        self,
        label_model = None,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
        support_list = None,
        sentence = None,
        prompts = None
    ):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()


        label_list = list()
        if label_model == None:
            for support in support_list:
                label_list.append(self.get_label(**support))
        else:
            for support in support_list:
                label_list.append(label_model.get_label(**support))

        # label_list = list()
        # for _ in support_list:
        #     label_list.append([self.tokenizer.vocab["Ġterrible"],self.tokenizer.vocab["Ġgreat"]])
        # support = support_list[0]
        # for _ in support_list:
        #     label_list.append(self.get_label(**support))
            # label_list.append([self.tokenizer.vocab["Ġanyway"],self.tokenizer.vocab["ĠAbsolutely"]])

        # Encode everything
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask
        )

        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.lm_head(sequence_mask_output)

        # Exit early and only return mask logits.
        # if self.return_full_softmax:
        #     if labels is not None:
        #         return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
        #     return prediction_mask_scores

        # Return logits for each label
        logits = []
        label_list = np.array(label_list)
        for label_id in range(len(label_list[0])):
            single_logits = list()
            for i, item in enumerate(prediction_mask_scores):
                single_logits.append(item[label_list[i][label_id]].to(device = labels.device))
            single_logits = torch.stack(single_logits,-1)
            logits.append(single_logits.unsqueeze(-1))
            # logits.append(torch.tensor(single_logits).unsqueeze(-1))
        logits = torch.cat(logits, -1).to(device = labels.device)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits) # Log prob of right polarity

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
        return ((loss,) + output) if loss is not None else output
