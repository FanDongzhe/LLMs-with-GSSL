import torch
import numpy as np

from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, IntervalStrategy
from Model import BertClassfier, BertClaInfModel, Evaluator

from sklearn.metrics import accuracy_score

from utils import init_path, time_logger
from torch.utils.data import random_split
from DataReader import Dataset


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy}


class LMTrainer():
    def __init__(self, seed, text, labels, filename):
        self.seed = seed

        self.model_name = "microsoft/deberta-base"
        self.feat_shrink = ""

        self.weight_decay = 0.0
        self.dropout = 0.3
        self.att_dropout = 0.1
        self.cla_dropout = 0.4
        self.batch_size = 9
        self.epochs = 4
        self.warmup_epochs = 0.6
        self.eval_patience = 50000
        self.grad_acc_steps = 1
        self.lr = 2e-5
        self.dataset_name = filename

        self.output_dir = f'/output/{self.dataset_name}/deberta-base'
        self.ckpt_dir = f'/prt_lm/{self.dataset_name}/deberta-base'


        self.text = text
        self.num_nodes = len(text)
        self.n_labels = len(set(labels))

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        X = tokenizer(text, padding=True, truncation=True, max_length=256)

        dataset = Dataset(X, labels)

        train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.inf_dataset = dataset




        bert_model = AutoModel.from_pretrained(self.model_name)
        self.model = BertClassfier(bert_model,
                                    n_labels=self.n_labels)
        self.model.config.dropout = self.dropout
        self.model.config.attention_dropout = self.att_dropout

        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)
        print(f"\nNumber of parameters: {trainable_params}")

    @time_logger
    def train(self):
        print(self.train_dataset)
        eq_batch_size = self.batch_size * 4
        train_steps = self.num_nodes // eq_batch_size + 1
        eval_steps = self.eval_patience // eq_batch_size
        warmup_steps = int(self.warmup_epochs * train_steps)

        args = TrainingArguments(
            output_dir=self.output_dir,
            do_train=True,
            do_eval=True,
            eval_steps=eval_steps,
            evaluation_strategy=IntervalStrategy.STEPS,
            save_steps=eval_steps,
            learning_rate=self.lr,
            weight_decay=self.weight_decay,
            save_total_limit=1,
            load_best_model_at_end=True,
            gradient_accumulation_steps=self.grad_acc_steps,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size*8,
            warmup_steps=warmup_steps,
            num_train_epochs=self.epochs,
            dataloader_num_workers=0,
            fp16=True,
            dataloader_drop_last=True,
        )
        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=compute_metrics,

        )

        self.trainer.train()
        torch.save(self.model.state_dict(), init_path(f"{self.ckpt_dir}.ckpt"))
        print(f'LM saved to {self.ckpt_dir}.ckpt')

    @time_logger
    @torch.no_grad()
    def eval_and_save(self):
        emb = np.memmap(init_path(f"{self.ckpt_dir}.emb"),
                        dtype=np.float16,
                        mode='w+',
                        shape=(self.num_nodes, 768))
        pred = np.memmap(init_path(f"{self.ckpt_dir}.pred"),
                         dtype=np.float16,
                         mode='w+',
                         shape=(self.num_nodes, self.n_labels))

        inf_model = BertClaInfModel(
            self.model, emb, pred)
        inf_model.eval()
        inference_args = TrainingArguments(
            output_dir=self.output_dir,
            do_train=False,
            do_predict=True,
            per_device_eval_batch_size=self.batch_size*8,
            dataloader_drop_last=False,
            dataloader_num_workers=0,
            fp16_full_eval=True,
        )

        trainer = Trainer(model=inf_model, args=inference_args)
        trainer.predict(self.inf_dataset)
        


    


