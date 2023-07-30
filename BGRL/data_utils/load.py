import os
import json
import torch
import csv
from data_utils.dataset import CustomDGLDataset
import numpy as np

def load_gpt_preds(dataset, topk):
    preds = []
    with open(f'gpt_preds/{dataset}.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            inner_list = []
            for value in row:
                inner_list.append(int(value))
            preds.append(inner_list)

    pl = torch.zeros(len(preds), topk, dtype=torch.long)
    for i, pred in enumerate(preds):
        pl[i][:len(pred)] = torch.tensor(pred[:topk], dtype=torch.long)+1
    return pl


def load_data_new(dataset, use_dgl=True, use_text=False, use_gpt=False, seed=0):
    if dataset == 'cora':
        from data_utils.load_cora import get_raw_text_cora as get_raw_text
    elif dataset == 'pubmed':
        from data_utils.load_pubmed import get_raw_text_pubmed as get_raw_text
    elif dataset == 'ogbn-arxiv':
        from data_utils.load_arxiv import get_raw_text_arxiv as get_raw_text
    else:
        exit(f'Error: Dataset {dataset} not supported')

    # for training GNN
    if not use_text:
        data, _ = get_raw_text(use_text=False, seed=seed)
        if use_dgl:
            data = CustomDGLDataset(dataset, data)
        return data

    # for finetuning LM
    if use_gpt:
        data, text = get_raw_text(use_text=False, seed=seed)
        folder_path = 'gpt_responses/{}'.format(dataset)
        print(f"using gpt: {folder_path}")
        n = data.y.shape[0]
        text = []
        for i in range(n):
            filename = str(i) + '.json'
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                content = json_data['choices'][0]['message']['content']
                text.append(content)
    else:
        data, text = get_raw_text(use_text=True, seed=seed)

    return data, text

def emb2dataX(emb_file_path):
    emb = np.memmap(emb_file_path, mode='r', dtype=np.float16, shape=(19717, 768))
    return torch.from_numpy(emb).to(torch.float32)

