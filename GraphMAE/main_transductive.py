import logging
import sys 
import numpy as np
from tqdm import tqdm
import torch
import os

from graphmae.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)
sys.path.append("..")
from graphmae.datasets.data_util import scale_feats
from data_utils.load import load_llm_feature_and_data
import data_utils.logistic_regression_eval as eval
from graphmae.models import build_model
from graphmae.evaluation import node_classification_evaluation

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def pretrain(model, graph, feat, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger=None):
    logging.info("start training..")
    graph = graph.to(device)
    x = feat.to(device)

    epoch_iter = tqdm(range(max_epoch))

    for epoch in epoch_iter:
        model.train()

        loss, loss_dict = model(graph, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)
        # break
        #if (epoch + 1) % 200 == 0:
        #    epacc = node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob, mute=True)

    # return best_model
    return model


def main(args):
    device = args.device if args.device >= 0 else "cpu"
    data_seeds = args.data_seeds
    model_seeds = args.model_seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate

    optim_type = args.optimizer 
    loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler


    graph = load_llm_feature_and_data(dataset_name=args.dataset,LLM_feat_seed=model_seeds[0],lm_model_name='microsoft/deberta-base',
                               feature_type=args.feature_type, use_dgl = True , device = args.device , 
                               sclae_feat= True if dataset_name == "ogbn-arxiv" else False )
    
    #graph.ndata['feat'] = scale_feats(graph.ndata['feat'].cpu()).to(args.device) # !the GraphMAE scaled feat
    # graph.ndata['feat'] = scale_feats(graph.ndata['feat'].cpu()).to('cpu')
    features = graph.ndata['feat'] 
    
    # graph, (num_features, num_classes) = load_dataset(dataset_name)
    
    (num_features, num_classes)= (features.shape[1],graph.ndata['label'].unique().size(0))
    args.num_features = num_features

    final_acc_list = []
    early_stp_acc_list=[]
    for i, seed in enumerate(model_seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None

        model = build_model(args)
        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None

        x = features

        if not load_model:
            model = pretrain(model, graph, x, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger)
            model = model.cpu()

        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load('C:/Users/YI/Desktop/cora_checkpoint.pt'))
        if save_model:
            logging.info("Saveing Model ...")
            torch.save(model.state_dict(), "checkpoint.pt")

        model = model.to(device)
        model.eval()


        #acc_list = node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f,max_epoch_f, device,dataset_name=args.dataset,data_random_seeds=args.data_seeds,mute=False)
        with torch.no_grad():
            feat = model.embed(graph.to(device), x.to(device))
            in_feat = x.shape[1]
        
        
        final_acc , early_stp_acc = eval.fit_logistic_regression_new(features=feat,labels=graph.ndata['label'],data_random_seeds=args.data_seeds,dataset_name=args.dataset,device=device)
        final_acc_list.extend(final_acc)
        early_stp_acc_list.extend(early_stp_acc)
        if logger is not None:
            logger.finish()

    final_acc, final_acc_std = np.mean(final_acc_list), np.std(final_acc_list)
    estp_acc, estp_acc_std = np.mean(early_stp_acc_list), np.std(early_stp_acc_list)
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    print(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.4f}")

    # mean_score = final_acc
    # std_score = final_acc_std
    # Ensure the directory exists
    # os.makedirs(args.logdir, exist_ok=True)

    # filename = f"final_score.txt"
    # with open(os.path.join(args.logdir, filename), 'w') as f:
    #     f.write(f"Mean: {mean_score}\n")
    #     f.write(f"Standard Deviation: {std_score}\n")
    # print(f"Final Score - Mean: {mean_score}, Standard Deviation: {std_score}")


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    print(args)
    main(args)
