import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import random
from models import DGI, LogReg
from utils import process
import pdb
import aug
import os
import argparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold
import json

parser = argparse.ArgumentParser("My DGI")

parser.add_argument('--dataset',          type=str,           default="",                help='data')
parser.add_argument('--aug_type',         type=str,           default="",                help='aug type: mask or edge')
parser.add_argument('--drop_percent',     type=float,         default=0.1,               help='drop percent')
parser.add_argument('--seed',             type=int,           default=39,                help='seed')
parser.add_argument('--gpu',              type=int,           default=0,                 help='gpu')
parser.add_argument('--save_name',        type=str,           default='try.pkl',                help='save ckpt name')
parser.add_argument('--k_shot',        type=float,           default='5',                help='number of samples per class')

args = parser.parse_args()

print('-' * 100)
print(args)
print('-' * 100)

dataset = args.dataset
aug_type = args.aug_type
drop_percent = args.drop_percent
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# training params
batch_size = 1
nb_epochs = 2
patience = 20
lr = 0.001
l2_coef = 0.0
drop_prob = 0.0
hid_units = 512
sparse = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nonlinearity = 'prelu' # special name to separate parameters
#adj, features, labels, idx_train, idx_val, idx_test,nb_nodes,ft_size,nb_classes = process.load_data(dataset)
adj, features, labels, idx_train, idx_val, idx_test,nb_nodes,ft_size = process.load_data_new(dataset)

features = process.preprocess_features(features)
#features = torch.FloatTensor(features[np.newaxis])

def one_hot_to_label(one_hot_encoding):
    return np.argmax(one_hot_encoding, axis=1)
labels_eavl = one_hot_to_label(labels)



'''
------------------------------------------------------------
edge node mask subgraph
------------------------------------------------------------
'''
print("Begin Aug:[{}]".format(args.aug_type))
if args.aug_type == 'edge':

    aug_features1 = features
    aug_features2 = features

    aug_adj1 = aug.aug_random_edge(adj, drop_percent=drop_percent) # random drop edges
    aug_adj2 = aug.aug_random_edge(adj, drop_percent=drop_percent) # random drop edges

elif args.aug_type == 'node':

    aug_features1, aug_adj1 = aug.aug_drop_node(features, adj, drop_percent=drop_percent)
    aug_features2, aug_adj2 = aug.aug_drop_node(features, adj, drop_percent=drop_percent)

elif args.aug_type == 'subgraph':

    aug_features1, aug_adj1 = aug.aug_subgraph(features, adj, drop_percent=drop_percent)
    aug_features2, aug_adj2 = aug.aug_subgraph(features, adj, drop_percent=drop_percent)

elif args.aug_type == 'mask':

    aug_features1 = aug.aug_random_mask(features,  drop_percent=drop_percent)
    aug_features2 = aug.aug_random_mask(features,  drop_percent=drop_percent)

    aug_adj1 = adj
    aug_adj2 = adj

else:
    assert False



'''
------------------------------------------------------------
'''

adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
aug_adj1 = process.normalize_adj(aug_adj1 + sp.eye(aug_adj1.shape[0]))
aug_adj2 = process.normalize_adj(aug_adj2 + sp.eye(aug_adj2.shape[0]))

if sparse:
    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
    sp_aug_adj1 = process.sparse_mx_to_torch_sparse_tensor(aug_adj1)
    sp_aug_adj2 = process.sparse_mx_to_torch_sparse_tensor(aug_adj2)

else:
    adj = (adj + sp.eye(adj.shape[0])).todense()
    aug_adj1 = (aug_adj1 + sp.eye(aug_adj1.shape[0])).todense()
    aug_adj2 = (aug_adj2 + sp.eye(aug_adj2.shape[0])).todense()


'''
------------------------------------------------------------
mask
------------------------------------------------------------
'''

'''
------------------------------------------------------------
'''
if not sparse:
    adj = torch.FloatTensor(adj[np.newaxis])
    aug_adj1 = torch.FloatTensor(aug_adj1[np.newaxis])
    aug_adj2 = torch.FloatTensor(aug_adj2[np.newaxis])

labels = torch.FloatTensor(labels[np.newaxis])
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)


model = DGI(ft_size, hid_units, nonlinearity)
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

if torch.cuda.is_available():
    print('Using CUDA')
    model.cuda()
    features = features.cuda()
    aug_features1 = aug_features1.cuda()
    aug_features2 = aug_features2.cuda()
    if sparse:
        sp_adj = sp_adj.cuda()
        sp_aug_adj1 = sp_aug_adj1.cuda()
        sp_aug_adj2 = sp_aug_adj2.cuda()
    else:
        adj = adj.cuda()
        aug_adj1 = aug_adj1.cuda()
        aug_adj2 = aug_adj2.cuda()

    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    #idx_test = idx_test.to(device)

b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 1e9
best_t = 0

for epoch in range(nb_epochs):

    model.train()
    optimiser.zero_grad()

    idx = np.random.permutation(nb_nodes)
    shuf_fts = features[:, idx, :]

    lbl_1 = torch.ones(batch_size, nb_nodes)
    lbl_2 = torch.zeros(batch_size, nb_nodes)
    lbl = torch.cat((lbl_1, lbl_2), 1)

    if torch.cuda.is_available():
        shuf_fts = shuf_fts.cuda()
        lbl = lbl.cuda()

    logits = model(features, shuf_fts, aug_features1, aug_features2,
                   sp_adj if sparse else adj,
                   sp_aug_adj1 if sparse else aug_adj1,
                   sp_aug_adj2 if sparse else aug_adj2,
                   sparse, None, None, None, aug_type=aug_type)

    loss = b_xent(logits, lbl)
    print('Loss:[{:.4f}]'.format(loss.item()))

    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
        torch.save(model.state_dict(), args.save_name)
    else:
        cnt_wait += 1

    if cnt_wait == patience:
        print('Early stopping!')
        break

    loss.backward()
    optimiser.step()

print('Loading {}th epoch'.format(best_t))
model.load_state_dict(torch.load(args.save_name))

embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None)
accs = []
accs = eval.fit_logistic_regression(embeds[0,].cpu().numpy(),labels_eavl,args.dataset)

#打印accs数组的均值和方差
print('Test acc:[{:.4f}]'.format(np.mean(accs)))
print('Test std:[{:.4f}]'.format(np.std(accs)))


