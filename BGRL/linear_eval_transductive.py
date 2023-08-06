import copy
import logging
import os

from absl import app
from absl import flags
import torch
from torch.nn.functional import cosine_similarity
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
#from torch_geometric.utils.sparse import to_edge_index
import json
from bgrl import *
from bgrl import BGRL
import sys
sys.path.append("..") 
from data_utils.load import load_llm_feature_and_data
import data_utils.logistic_regression_eval as eval

log = logging.getLogger(__name__)
FLAGS = flags.FLAGS
# Dataset.
flags.DEFINE_enum('dataset', 'cora',
                  ['cora', 'Citeseer', 'Pubmed'],
                  'Which graph dataset to use.')
flags.DEFINE_string('dataset_dir', './data', 'Where the dataset resides.')

# Architecture.
flags.DEFINE_multi_integer('graph_encoder_layer', None, 'Conv layer sizes.')
flags.DEFINE_string('ckpt_path', None, 'Path to checkpoint.')


def main(argv):
    # use CUDA_VISIBLE_DEVICES to select gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    log.info('Using {} for evaluation.'.format(device))

    # load data
    if FLAGS.dataset == 'reddit':
        dataset, train_masks, val_masks, test_masks = get_reddit(FLAGS.dataset_dir)
    elif FLAGS.dataset in ('cora', 'citeseer', 'pubmed'):
        dataset, train_masks, val_masks, test_masks = get_dataset(FLAGS.dataset_dir, FLAGS)
        print(train_masks)
    elif FLAGS.dataset == 'corafull':
        dataset, train_masks, val_masks, test_masks = get_corafull(FLAGS.dataset_dir)
    elif FLAGS.dataset == 'photo':
        dataset, train_masks, val_masks, test_masks = get_amazonphoto(FLAGS.dataset_dir)
    elif FLAGS.dataset == 'computer':
        dataset, train_masks, val_masks, test_masks = get_amazoncomputer(FLAGS.dataset_dir)
    elif FLAGS.dataset == 'ogbn-arxiv':
        dataset, train_masks, val_masks, test_masks = get_ogbn_arxiv(FLAGS.dataset_dir)

    data = dataset[0]  # all dataset include one graph
    log.info('Dataset {}, {}.'.format(dataset.__class__.__name__, data))
    data = data.to(device)  # permanently move in gpy memory

    # build networks
    input_size, representation_size = data.x.size(1), FLAGS.graph_encoder_layer[-1]
    encoder = GCN([input_size] + FLAGS.graph_encoder_layer, batchnorm=True) # 512, 256, 128
    load_trained_encoder(encoder, FLAGS.ckpt_path, device)
    encoder.eval()

    # compute representations
    representations, labels = compute_representations(encoder, dataset, device)

    # fit logistic regression
    scores = eval.fit_logistic_regression(representations.cpu().numpy(), labels.cpu().numpy(),FLAGS.dataset,FLAGS.number_of_splits)

    score = np.mean(scores)
    print('Test score: %.5f' %score)


if __name__ == "__main__":
    log.info('PyTorch version: %s' % torch.__version__)
    app.run(main)
