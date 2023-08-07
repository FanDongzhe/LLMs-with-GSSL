import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder, normalize
from sklearn.model_selection import StratifiedKFold
import torch
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim as optim

def split_data_k(y, k_shot=20, data_random_seed=0):
    np.random.seed(data_random_seed)
    num_classes = y.max() + 1
    all_indices = np.arange(len(y))

    train_indices = []

    for i in range(num_classes):
        class_indices = np.where(y == i)[0]
        if len(class_indices) < k_shot:
            raise ValueError(f"Not enough samples in class {i} for k-shot learning")
        class_train_indices = np.random.choice(class_indices, k_shot, replace=False)
        train_indices.extend(class_train_indices)

    all_indices = np.setdiff1d(all_indices, train_indices)

    val_indices = []

    for i in range(num_classes):
        class_indices = np.where(y == i)[0]
        class_indices = np.setdiff1d(class_indices, train_indices)  # remove already chosen train_indices
        class_val_indices = np.random.choice(class_indices, 30, replace=False)
        val_indices.extend(class_val_indices)

    val_indices = np.array(val_indices)
    all_indices = np.setdiff1d(all_indices, val_indices)

    # All remaining indices will be for testing
    test_indices = all_indices

    train_mask = np.isin(np.arange(len(y)), train_indices)
    val_mask = np.isin(np.arange(len(y)), val_indices)
    test_mask = np.isin(np.arange(len(y)), test_indices)

    return train_mask, val_mask, test_mask


def fit_logistic_regression(X, y, dataset_name,data_random_seeds):
    one_hot_encoder = OneHotEncoder(categories='auto', sparse=False)

    y_one_hot = one_hot_encoder.fit_transform(y.reshape(-1, 1)).astype(bool)

    X = normalize(X, norm='l2')

    accuracies = []
    for data_random_seed in data_random_seeds:
        if dataset_name in ('Cora','Pubmed','cora','pubmed'):
            train_mask, val_mask, test_mask = split_data_k(y, k_shot=20,data_random_seed=data_random_seed)
            X_train, y_train = X[train_mask], y_one_hot[train_mask]
            X_val, y_val = X[val_mask], y_one_hot[val_mask]
            X_test, y_test = X[test_mask], y_one_hot[test_mask]
        else:
            rng = np.random.RandomState(data_random_seed)  # this will ensure the dataset will be split exactly the same
            # throughout training
            #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=rng)
            X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.8, random_state=rng)

        logreg = LogisticRegression(solver='liblinear')
        c = 2.0 ** np.arange(-10, 11)
        #cv = ShuffleSplit(n_splits=5, test_size=0.5)
        cv = StratifiedKFold(n_splits=2)

        clf = GridSearchCV(estimator=OneVsRestClassifier(logreg), param_grid=dict(estimator__C=c),
                           n_jobs=5, cv=cv, verbose=0)

        #clf.fit(X_train, y_train)
        y_train_labels = np.argmax(y_train, axis=1)
        clf.fit(X_train, y_train_labels)

        y_pred = clf.predict_proba(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(bool)

        test_acc = metrics.accuracy_score(y_test, y_pred)
        accuracies.append(test_acc)
    return accuracies

class LogisticRegression_nn(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)

    def forward(self, g, x, *args):
        logits = self.linear(x)
        return logits

def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()

    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer



def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)

def fit_logistic_regression_new(data,dataset_name,data_random_seeds,device='cpu'):
    in_feat = data.x.shape[1]
    if hasattr(data, 'y') and data.y is not None:
        num_classes = data.y.max().item() + 1
    else:
        num_classes = data.ndata['label'].max().item() + 1
    encoder = LogisticRegression_nn(in_feat, num_classes)
    encoder.to(device)
    optimizer_f = create_optimizer("adam", encoder, lr=0.01, weight_decay=1e-4)
    accs = linear_probing_for_transductive_node_classiifcation(encoder, data, optimizer_f,data_random_seeds=data_random_seeds, dataset_name=dataset_name,
                                                                              max_epoch=200, device=device, mute=False)
    return accs


def linear_probing_for_transductive_node_classiifcation(model, graph, optimizer, max_epoch, data_random_seeds, dataset_name, device, mute=False):
    criterion = torch.nn.CrossEntropyLoss()

    graph = graph.to(device)
    x = graph.x.to(device)

    if hasattr(graph, 'y') and graph.y is not None:
        labels = graph.y
    else:
        labels = graph.ndata['label']

    accs = []
    for data_random_seed in data_random_seeds:
        if dataset_name in ('Pubmed','pubmed'):
            train_mask, val_mask, test_mask = split_data_k(labels, k_shot=20,data_random_seed=data_random_seed)
        else:
            rng = np.random.RandomState(data_random_seed)  # this will ensure the dataset will be split exactly the same
            indices = np.arange(len(x))
            #train：0.2   val：0.2   test：0.6
            train_indices, temp_indices, y_train, y_temp = train_test_split(indices, labels, test_size=0.8, random_state=rng)
            val_indices, test_indices, y_val, y_test = train_test_split(temp_indices, y_temp, test_size=0.75,random_state=rng)
            # Create train_mask, val_mask, and test_mask
            train_mask = np.zeros(len(x), dtype=bool)
            val_mask = np.zeros(len(x), dtype=bool)
            test_mask = np.zeros(len(x), dtype=bool)
            train_mask[train_indices] = True
            val_mask[val_indices] = True
            test_mask[test_indices] = True



        best_val_acc = 0
        best_val_epoch = 0
        best_model = None

        if not mute:
            epoch_iter = tqdm(range(max_epoch))
        else:
            epoch_iter = range(max_epoch)

        for epoch in epoch_iter:
            model.train()
            out = model(graph, x)
            loss = criterion(out[train_mask], labels[train_mask])
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
            optimizer.step()

            with torch.no_grad():
                model.eval()
                pred = model(graph, x)
                val_acc = accuracy(pred[val_mask], labels[val_mask])
                val_loss = criterion(pred[val_mask], labels[val_mask])
                test_acc = accuracy(pred[test_mask], labels[test_mask])
                test_loss = criterion(pred[test_mask], labels[test_mask])

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_val_epoch = epoch
                best_model = copy.deepcopy(model)

            if not mute:
                epoch_iter.set_description(
                    f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_acc:{val_acc}, test_loss:{test_loss.item(): .4f}, test_acc:{test_acc: .4f}")

        best_model.eval()
        with torch.no_grad():
            pred = best_model(graph, x)
            estp_test_acc = accuracy(pred[test_mask], labels[test_mask])
        if mute:
            print(
                f"# IGNORE: --- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")
        else:
            print(
                f"--- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")

        accs.append(estp_test_acc)
    # (final_acc, es_acc, best_acc)
    return accs
