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
from LLMs.EmbTrainer import LMTrainer
import data_utils.logistic_regression_eval as evaluate
log = logging.getLogger(__name__)
FLAGS = flags.FLAGS
flags.DEFINE_multi_integer('model_seeds', [0], 'Random seed used to generate train/val/test split.')
flags.DEFINE_multi_integer('data_seeds', [1], 'Random seed used to generate train/val/test split.')
#flags.DEFINE_integer('num_eval_splits', 2, 'Number of different train/test splits the model will be evaluated over.')

# Dataset.
flags.DEFINE_enum('dataset', 'cora',
                  ['cora',  'pubmed','ogbn-arxiv','amazon-photo','amazon-computers',"amazon-history"],
                  'Which graph dataset to use.')
flags.DEFINE_string('dataset_dir', './data', 'Where the dataset resides.')
flags.DEFINE_string('feature_type', 'TA', 'LLM feature type')

# Architecture.
flags.DEFINE_multi_integer('graph_encoder_layer', None, 'Conv layer sizes.')
flags.DEFINE_integer('predictor_hidden_size', 512, 'Hidden size of projector.')

# Training hyperparameters.
flags.DEFINE_integer('epochs', 1000, 'The number of training epochs.')
flags.DEFINE_integer('device', 0, 'GPU index')

flags.DEFINE_float('lr', 1e-5, 'The learning rate for model training.')
flags.DEFINE_float('weight_decay', 1e-5, 'The value of the weight decay for training.')
flags.DEFINE_float('mm', 0.99, 'The momentum for moving average.')
flags.DEFINE_integer('lr_warmup_epochs', 1000, 'Warmup period for learning rate.')

# Augmentations.
flags.DEFINE_float('drop_edge_p_1', 0., 'Probability of edge dropout 1.')
flags.DEFINE_float('drop_feat_p_1', 0., 'Probability of node feature dropout 1.')
flags.DEFINE_float('drop_edge_p_2', 0., 'Probability of edge dropout 2.')
flags.DEFINE_float('drop_feat_p_2', 0., 'Probability of node feature dropout 2.')

# Logging and checkpoint.
flags.DEFINE_string('logdir', None, 'Where the checkpoint and logs are stored.')

# Evaluation
flags.DEFINE_integer('eval_epochs', 25, 'Evaluate every eval_epochs.')


#dataset setting
flags.DEFINE_bool('use_dgl', False, '.')
flags.DEFINE_bool('use_text', False, '.')
flags.DEFINE_bool('use_gpt', False, '.')
flags.DEFINE_bool('use_BoW', True, '.')

flags.DEFINE_bool('use_LLM_emb', False, '.')
flags.DEFINE_bool('train_both', False, '.')

def main(argv):
    final_acc_list = []
    early_stp_acc_list=[]
    for model_seed in FLAGS.model_seeds:
        # use CUDA_VISIBLE_DEVICES to select gpu
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        log.info('Using {} for training.'.format(device))

        # set random seed
        if model_seed is not None:
            log.info('Random seed set to {}.'.format(model_seed))
            set_random_seeds(random_seed=model_seed)

        # create log directory
        os.makedirs(FLAGS.logdir, exist_ok=True)
        with open(os.path.join(FLAGS.logdir, 'config.cfg'), "w") as file:
            file.write(FLAGS.flags_into_string())  # save config file


        # load data
        dataset = load_llm_feature_and_data(
            dataset_name = FLAGS.dataset,
            lm_model_name='microsoft/deberta-base',
            feature_type=FLAGS.feature_type,
            device=device,
            use_text=FLAGS.use_LLM_emb,
            )

        if FLAGS.dataset == 'ogbn-arxiv':
            dataset.edge_index,_ = to_edge_index(dataset.edge_index)

        data = dataset
        if FLAGS.use_LLM_emb:
            # Code from trainModel.py to get emb
            text = data.text
            labels = data.y.tolist()
            seed = 0
            trainer_LM = LMTrainer(seed, text, labels, FLAGS.dataset)
            trainer_LM.train()
            emb, _ = trainer_LM.eval_and_save()
            data.x = torch.from_numpy(emb).to(torch.float32)
        #data.x = data.x.float()

        data = data.to(device)  # permanently move in gpy memory

        # prepare transforms
        transform_1 = get_graph_drop_transform(drop_edge_p=FLAGS.drop_edge_p_1, drop_feat_p=FLAGS.drop_feat_p_1)
        transform_2 = get_graph_drop_transform(drop_edge_p=FLAGS.drop_edge_p_2, drop_feat_p=FLAGS.drop_feat_p_2)

        # build networks
        input_size, representation_size = data.x.size(1), FLAGS.graph_encoder_layer[-1]
        encoder = GCN([input_size] + FLAGS.graph_encoder_layer, batchnorm=True)   # 512, 256, 128
        predictor = MLP_Predictor(representation_size, representation_size, hidden_size=FLAGS.predictor_hidden_size)
        model: BGRL = BGRL(encoder, predictor).to(device)

        # optimizer
        # Combined trainable parameters from both models
        if FLAGS.train_both:
            combined_parameters = list(model.trainable_parameters()) + list(trainer_LM.model.parameters())
            optimizer = AdamW(combined_parameters, lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
        else:
            optimizer = AdamW(model.trainable_parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)

        # scheduler
        lr_scheduler = CosineDecayScheduler(FLAGS.lr, FLAGS.lr_warmup_epochs, FLAGS.epochs)
        mm_scheduler = CosineDecayScheduler(1 - FLAGS.mm, 0, FLAGS.epochs)

        # setup tensorboard and make custom layout
        writer = SummaryWriter(FLAGS.logdir)
        #layout = {'accuracy': {'accuracy/test': ['Multiline', [f'accuracy/test_{i}' for i in range(num_eval_splits)]]}}
        #writer.add_custom_scalars(layout)

        def train(step):
            model.train()

            # update learning rate
            lr = lr_scheduler.get(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # update momentum
            mm = 1 - mm_scheduler.get(step)

            # forward
            optimizer.zero_grad()

            x1, x2 = transform_1(data), transform_2(data)

            q1, y2 = model(x1, x2)
            q2, y1 = model(x2, x1)

            loss = 2 - cosine_similarity(q1, y2.detach(), dim=-1).mean() - cosine_similarity(q2, y1.detach(), dim=-1).mean()
            # Training code from trainModel.py
            if FLAGS.train_both:
                trainer_LM.train()
                torch.save(trainer_LM.model.state_dict(), "model_checkpoint.ckpt")
                # Combined loss from both models
                combined_loss = loss  # You can adjust this as per requirement
                combined_loss.backward()
            else:
                loss.backward()


            # update online network
            optimizer.step()
            # update target network
            model.update_target_network(mm)

            # log scalars
            writer.add_scalar('params/lr', lr, step)
            writer.add_scalar('params/mm', mm, step)
            writer.add_scalar('train/loss', loss, step)

        def eval(epoch):
            # make temporary copy of encoder
            tmp_encoder = copy.deepcopy(model.online_encoder).eval()
            representations, labels = compute_representations(tmp_encoder, dataset, device)

            # evaluate
            #scores = fit_logistic_regression(representations.cpu().numpy(), labels.cpu().numpy(),FLAGS.dataset, data_random_seeds = FLAGS.data_seeds)
            #scores = fit_logistic_regression_new(data,FLAGS.dataset, data_random_seeds = FLAGS.data_seeds)

            final_acc, early_stp_acc = evaluate.fit_logistic_regression_new(features=representations, labels=labels,
                                                                        data_random_seeds=FLAGS.data_seeds,
                                                                        dataset_name=FLAGS.dataset, device=device,mute=True)

            for i, score in enumerate(early_stp_acc):
                writer.add_scalar(f'accuracy/test_{i}', score, epoch)
                #print(f'Score {i}: {score}')
            return final_acc,early_stp_acc

        # train and evaluate
        for epoch in tqdm(range(1, FLAGS.epochs + 1)):
            train(epoch-1)

        # save encoder weights
        torch.save({'model': model.online_encoder.state_dict()}, os.path.join(FLAGS.logdir, f'{FLAGS.dataset}.pt'))
        final_acc,early_stp_acc = eval(99999)
        final_acc_list.append(final_acc)
        early_stp_acc_list.append(early_stp_acc)


    final_acc, final_acc_std = np.mean(final_acc_list), np.std(final_acc_list)
    estp_acc, estp_acc_std = np.mean(early_stp_acc_list), np.std(early_stp_acc_list)
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    print(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.4f}")




if __name__ == "__main__":
    log.info('PyTorch version: %s' % torch.__version__)
    app.run(main)

