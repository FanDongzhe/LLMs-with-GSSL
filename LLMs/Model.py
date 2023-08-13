import torch
import torch.nn as nn
import numpy as np
from transformers import PreTrainedTokenizer, PreTrainedModel, AutoModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import TokenClassifierOutput

class BertClassfier(PreTrainedModel):
    def __init__(self, model, n_labels, dropout=0.0, seed=0, cla_bias=True):
        super().__init__(model.config)
        self.bert = model
        self.dropout = nn.Dropout(dropout)
        hidden_sim = model.config.hidden_size
        self.loss_func = nn.CrossEntropyLoss(label_smoothing=0.3, reduction="mean")

        self.classfier = nn.Linear(hidden_sim, n_labels, bias=cla_bias)


    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                return_dict=None,
                preds=None):
        
        bert_output = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask,
                                return_dict=return_dict,
                                output_hidden_states=True)
        
        emb = self.dropout(bert_output['hidden_states'][-1])

        cls_token_emb = emb.permute(1,0,2)[0]


        logits = self.classfier(cls_token_emb)

        if labels.shape[-1] == 1:
            labels = labels.squeeze()
        loss = self.loss_func(logits, labels)

        return TokenClassifierOutput(loss=loss, logits=logits)
    
class BertClaInfModel(PreTrainedModel):
    def __init__(self, model, emb, pred):
        super().__init__(model.config)
        self.bert_classfier = model
        self.emb, self.pred = emb, pred
        self.loss_func = nn.CrossEntropyLoss(label_smoothing=0.3,
                                             reduction='mean')
        
    @torch.no_grad()
    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                return_dict=None,
                node_id=None):
        
        bert_output = self.bert_classfier.bert(input_ids=input_ids,
                                              attention_mask=attention_mask,
                                              return_dict=return_dict,
                                              output_hidden_states=True)
        
        emb = bert_output['hidden_states'][-1]

        cls_token_emb = emb.permute(1,0,2)[0]

        logits = self.bert_classfier.classifier(cls_token_emb)

        batch_nodes = node_id.cpu().numpy()
        self.emb[batch_nodes] = cls_token_emb.cpu().numpy().astype(np.float16)
        self.pred[batch_nodes] = logits.cpu().numpy().astype(np.float16)

        if labels.shape[-1] == 1:
            labels = labels.squeeze()
        loss = self.loss_func(logits, labels)

        return TokenClassifierOutput(loss=loss, logits=logits)
    
class Evaluator:

    def __init__(self, name):
        self.name = name

    def eval(self, input_dict):
        y_true, y_pred = input_dict["y_true"], input_dict["y_pred"]
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        acc_list = []

        for i in range(y_true.shape[1]):
            is_labeled = y_true[:, i] == y_true[:, i]
            correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
            acc_list.append(float(np.sum(correct))/len(correct))

        return {'acc': sum(acc_list)/len(acc_list)}
        


        