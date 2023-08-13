import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
#from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import torch
import torch.nn as nn
import torch
import random
from ogb.nodeproppred import DglNodePropPredDataset
from torch_geometric import datasets
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected
import re
import dgl
import numpy as np
from scipy.sparse import csr_matrix
import torch_geometric
from scipy.sparse import lil_matrix

# from data_utils.load import load_data as load_data_text
# from data_utils.load import emb2dataX

sys.path.append("..") 
from data_utils.load import load_llm_feature_and_data
###############################################
# This section of code adapted from tkipf/gcn #
###############################################

from seed import seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# 应用其他需要种子的库或模块

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)




def load_data(dataset_str,device,feature_type,):
    if dataset_str in ['cora','pubmed'] or "amazon" in dataset_str:
        g = load_llm_feature_and_data(dataset_name=dataset_str,LLM_feat_seed=0,lm_model_name='microsoft/deberta-base',
                               feature_type=feature_type, use_dgl = True , device = device ).cpu()
        # 获取图数据
        
        # 将DGL图转换为CSR格式的邻接矩阵
        # adjacency = g.adjacency_matrix_scipy(return_edge_ids=False)
        adjacency = g.adj_external(scipy_fmt='csr')
        adj = csr_matrix(adjacency)
        # 将DGL图的节点特征转换为LIL格式的矩阵
        features_ = g.ndata['feat']
        features = lil_matrix(features_)
        # 获取标签矩阵
        labels = g.ndata['label']
        labels_np = labels.numpy()
        nb_classes = labels_np.max() + 1
        one_hot_labels = np.zeros((labels_np.size, nb_classes))
        one_hot_labels[np.arange(labels_np.size), labels_np] = 1
        labels = one_hot_labels
        # 获取训练、验证和测试节点的索引
        # idx_train = np.where(g.ndata['train_mask'])[0]
        # idx_val = np.where(g.ndata['val_mask'])[0]
        # idx_test = np.where(g.ndata['test_mask'])[0]

        # 获取节点数量、特征大小和类别数量
        nb_nodes = g.number_of_nodes()
        ft_size = features_.shape[1]
        nb_classes = g.ndata['label'].unique().size(0)
        return adj, features, labels, nb_nodes, ft_size, nb_classes

    elif dataset_str == 'ogbn-arxiv':
        g = load_llm_feature_and_data(dataset_name=dataset_str,LLM_feat_seed=0,lm_model_name='microsoft/deberta-base',
                               feature_type=feature_type, use_dgl = True , device = device ).cpu()
        
        # split_idx = dataset.get_idx_split()
        labels = g.ndata['label']  # 这是一个带有节点属性预测标签的图
        labels_np = labels.numpy()
        nb_classes = labels_np.max() + 1
        one_hot_labels = np.zeros((labels_np.size, nb_classes))
        one_hot_labels[np.arange(labels_np.size), labels_np] = 1
        labels = one_hot_labels
        # 将DGL图转换为CSR格式的邻接矩阵
        # adjacency = g.adjacency_matrix_scipy(return_edge_ids=False)
        adjacency = g.adj_external(scipy_fmt='csr')
        adj = csr_matrix(adjacency)

        # 将DGL图的节点特征转换为LIL格式的矩阵
        features_ = g.ndata['feat']
        features = lil_matrix(features_)


        # 获取训练、验证和测试节点的索引
        # idx_train = np.where(g.ndata['train_mask'])[0]
        # idx_val = np.where(g.ndata['val_mask'])[0]
        # idx_test = np.where(g.ndata['test_mask'])[0]

        # 获取节点数量、特征大小和类别数量
        nb_nodes = g.number_of_nodes()
        ft_size = features_.shape[1]
        nb_classes = g.ndata['label'].unique().size(0)
        return adj,features,labels, nb_nodes,ft_size,nb_classes
    else:
        raise ValueError(dataset_str)

def load_data_new(dataset_str):
    assert False
    if dataset_str in ['cora','pubmed']:
        if dataset_str == 'cora':
            dataset = load_data_text('cora', use_dgl=True, use_text=False, use_gpt=False, seed=0)
        if dataset_str == 'pubmed':
            dataset = load_data_text('pubmed', use_dgl=True, use_text=False, use_gpt=False, seed=0)
                
    
        dataset_old = dgl.data.PubmedGraphDataset()
        # 获取图数据
        g = dataset[0]
        # 将DGL图转换为CSR格式的邻接矩阵
        adjacency = g.adjacency_matrix_scipy(return_edge_ids=False)
        adj = csr_matrix(adjacency)
        # 将DGL图的节点特征转换为LIL格式的矩阵
        features_ = g.ndata['feat']
        features = lil_matrix(features_)
        # 获取标签矩阵
        labels = g.ndata['label']
        labels_np = labels.numpy()
        nb_classes = labels_np.max() + 1
        one_hot_labels = np.zeros((labels_np.size, nb_classes))
        one_hot_labels[np.arange(labels_np.size), labels_np] = 1
        labels = one_hot_labels
        # 获取训练、验证和测试节点的索引
        idx_train = None
        idx_val = None
        idx_test = None

        # 获取节点数量、特征大小和类别数量
        nb_nodes = g.number_of_nodes()
        ft_size = features_.shape[1]
        if dataset_str == 'pubmed':
            emb = emb2dataX('./prt_lm/pubmed/microsoft/deberta-base-seed0.emb')
        features = emb
        return adj, features, labels, idx_train, idx_val, idx_test, nb_nodes, ft_size

    if dataset_str == 'ogbn-arxiv':
        dataset = DglNodePropPredDataset(name='ogbn-arxiv')
        g, labels = dataset[0]  # 这是一个带有节点属性预测标签的图
        labels = labels[:, 0]  # 取出标签

        # 将DGL图转换为CSR格式的邻接矩阵
        adjacency = g.adjacency_matrix_scipy(return_edge_ids=False)
        adj = csr_matrix(adjacency)

        # 将DGL图的节点特征转换为LIL格式的矩阵
        features_ = g.ndata['feat']
        features = lil_matrix(features_)

        # 获取训练、验证和测试节点的索引
        idx_train = None
        idx_val = None
        idx_test = None

        # 获取节点数量、特征大小和类别数量
        nb_nodes = g.number_of_nodes()
        ft_size = features_.shape[1]
        return adj,features,labels,idx_train, idx_val, idx_test, nb_nodes,ft_size


def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

