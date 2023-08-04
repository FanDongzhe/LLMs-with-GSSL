import os
import json
import torch
import csv
import numpy as np 
from data_utils.dataset import CustomDGLDataset


def load_gpt_preds(dataset, topk):
    preds = []
    with open(f'../gpt_preds/{dataset}.csv', 'r') as file:
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


def load_data(dataset, use_dgl=False, use_text=False, use_gpt=False, seed=0):
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

def load_llm_feature_and_data(dataset_name, feature_type, use_dgl = False, LLM_feat_seed = 0, lm_model_name="microsoft/deberta-base",
                              device = 0):
        '''
        args:
            feature_type: TA or E or P 
            lm_model_name: "microsoft/deberta-base"
            device: gpu index 
        
        returns : 
        
        note : remove the seed for load_data since we will unify the split 
        
        '''
        seed = LLM_feat_seed
        # ! Load data from ogb
        data = load_data(dataset_name, use_dgl=use_dgl,
                         use_text=False)

        if  use_dgl:
            data=data[0]
            num_nodes = data.num_nodes()
            data.x = data.ndata['feat'] # ref https://github.com/XiaoxinHe/TAPE/blob/241c93b735dcebbe2853414395c1559d5c2ce202/core/GNNs/dgl_gnn_trainer.py#L39C8-L39C8
        else:
            num_nodes = data.x.shape[0]
            num_classes = data.y.unique().size(0)
            data.y = data.y.squeeze()

        # ! Init gnn feature
        topk = 3 if dataset_name == 'pubmed' else 5
        if feature_type == 'ogb':
            print("Loading OGB features...")
            features = data.x
        elif feature_type == 'TA':
            print("Loading pretrained LM features (title and abstract) ...")
            LM_emb_path = f"../prt_lm/{dataset_name}/{lm_model_name}-seed{seed}.emb"
            print(f"LM_emb_path: {LM_emb_path}")
            features = torch.from_numpy(np.array(
                np.memmap(LM_emb_path, mode='r',
                          dtype=np.float16,
                          shape=(num_nodes, 768)))
            ).to(torch.float32)
        elif feature_type == 'E':
            print("Loading pretrained LM features (explanations) ...")
            LM_emb_path = f"../prt_lm/{dataset_name}2/{lm_model_name}-seed{seed}.emb"
            print(f"LM_emb_path: {LM_emb_path}")
            features = torch.from_numpy(np.array(
                np.memmap(LM_emb_path, mode='r',
                          dtype=np.float16,
                          shape=(num_nodes, 768)))
            ).to(torch.float32)
        elif feature_type == 'P':
            print("Loading top-k prediction features ...")
            features = load_gpt_preds(dataset_name, topk)
            features=features.float()
        else:
            print(
                f'Feature type {feature_type} not supported. Loading OGB features...')
            feature_type = 'ogb'
            features = data.x
        if use_dgl:
            data.ndata['feat'] = features
        else:
            data.x = features.to(device)  # to fit dgl , use to() instead of .cuda()
        data = data.to(device)
        
        return data
                                  
def emb2dataX(emb_file_path):
    emb = np.memmap(emb_file_path, mode='r', dtype=np.float16, shape=(19717, 768))
    return torch.from_numpy(emb).to(torch.float32)
