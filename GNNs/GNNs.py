from sklearn.model_selection import train_test_split
from torch_geometric.nn import GCNConv, GINConv, GATConv
from data_utils.load import load_llm_feature_and_data
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import os
import sys

os.chdir(os.getcwd()+'\GNNs')
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, args.dim_hidden)
        self.conv2 = GCNConv(args.dim_hidden, num_classes)
        self.norm = torch.nn.BatchNorm1d(args.dim_hidden)

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.conv1(x, edge_index)
        x = self.norm(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.8, training=self.training)

        x = self.conv2(x, edge_index)
        return x

'''
class pair_norm(torch.nn.Module):
    def __init__(self):
        super(pair_norm, self).__init__()

    def forward(self, x):
        col_mean = x.mean(dim=0)
        x = x - col_mean
        rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
        x = x / rownorm_mean
        return x


class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.cached = self.transductive = args.transductive
        self.layers_GCN = nn.ModuleList([])
        self.layers_bn = nn.ModuleList([])
        if self.dataset == 'cora':
            self.num_feats = 1433
            self.num_classes = 7
        self.layers_GCN.append(GCNConv(self.num_feats, self.dim_hidden, cached=self.cached))
        self.layers_bn.append(torch.nn.BatchNorm1d(self.dim_hidden))


        for _ in range(self.num_layers - 2):
            self.layers_GCN.append(
                GCNConv(self.dim_hidden, self.dim_hidden, cached=self.cached))
            self.layers_bn.append(torch.nn.BatchNorm1d(self.dim_hidden))
        self.layers_GCN.append(GCNConv(self.dim_hidden, self.num_classes, cached=self.cached))

        self.layers_bn.append(torch.nn.BatchNorm1d(self.dim_hidden))
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=self.lr, weight_decay=self.weight_decay)

    def forward(self, x, edge_index):

        # implemented based on DeepGCN: https://github.com/LingxiaoShawn/PairNorm/blob/master/models.py

        for i in range(self.num_layers - 1):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.layers_GCN[i](x, edge_index)
            x = self.layers_bn[i](x)
            x = F.relu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers_GCN[-1](x, edge_index)
        return x
'''

class GIN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GIN, self).__init__()
        nn1 = torch.nn.Sequential(torch.nn.Linear(num_node_features, args.dim_hidden), torch.nn.ReLU(), torch.nn.Linear(args.dim_hidden,args.dim_hidden))
        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(torch.nn.Linear(args.dim_hidden, num_classes))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_node_features, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, num_classes, heads=8, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


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

def split_data_s(y,data_random_seed=0):
    # train：0.2   val：0.2   test：0.6
    rng = np.random.RandomState(data_random_seed)  # this will ensure the dataset will be split exactly the same
    indices = np.arange(len(y))
    train_indices, temp_indices, y_train, y_temp = train_test_split(indices, y, test_size=0.8, random_state=rng)
    val_indices, test_indices, y_val, y_test = train_test_split(temp_indices, y_temp, test_size=0.75, random_state=rng)
    # Create train_mask, val_mask, and test_mask
    train_mask = np.zeros(len(y), dtype=bool)
    val_mask = np.zeros(len(y), dtype=bool)
    test_mask = np.zeros(len(y), dtype=bool)
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    return train_mask, val_mask, test_mask


def train(model, data, device,dataseed):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    if args.dataset in ('Cora','PubMed','cora','pubmed'):
        train_mask, val_mask, test_mask = split_data_k(data.y, k_shot=20, data_random_seed=dataseed)
    else:
        train_mask, val_mask, test_mask = split_data_s(data.y, data_random_seed=dataseed)
    best_val_accuracy = 0.0
    model.train()
    for epoch in range(10):
        '''
        out = model(data.x,data.edge_index)
        model.optimizer.zero_grad()
        loss = loss_function(out[train_mask], data.y[train_mask])
        loss.backward()
        model.optimizer.step()
        '''
        out = model(data)
        optimizer.zero_grad()
        loss = loss_function(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        # Evaluate on validation data
        val_accuracy = evaluate(model, data, val_mask)
        print('Epoch {:03d} loss {:.4f}, Val Accuracy: {:.4f}'.format(epoch, loss.item(), val_accuracy))

        #Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = model.state_dict()  # Save the model parameters

    #Load the best model for testing
    model.load_state_dict(best_model)
    return test_mask

def evaluate(model, data, val_mask):
    model.eval()
    #_, pred = model(data.x,data.edge_index).max(dim=1)
    _, pred = model(data).max(dim=1)
    correct = int(pred[val_mask].eq(data.y[val_mask]).sum().item())
    acc = correct / int(val_mask.sum())
    #print('Val Accuracy: {:.4f}'.format(acc))
    return acc

def test(model, data, test_mask):
    model.eval()
    #_, pred = model(data.x,data.edge_index).max(dim=1)
    _, pred = model(data).max(dim=1)
    correct = int(pred[test_mask].eq(data.y[test_mask]).sum().item())
    acc = correct / int(test_mask.sum())
    print('TEST Accuracy: {:.4f}'.format(acc))
    return acc


def main(args):

    dataset = load_llm_feature_and_data(
        dataset_name=args.dataset,
        lm_model_name='microsoft/deberta-base',
        feature_type=args.feature_type,
        device='cpu')

    num_node_features = dataset.x.shape[1]
    num_classes =dataset.y.max().item() + 1

    device = args.device

    accs = []
    dataseeds = np.random.choice(1000, 10, replace=False)
    for dataseed in dataseeds:
        if args.model_type == 'GCN':
            #model = GCN(args).to('cpu')
            model = GCN(num_node_features, num_classes).to('cpu')
        elif args.model_type == 'GIN':
            model = GIN(num_node_features, num_classes).to('cpu')
        elif args.model_type == 'GAT':
            model = GAT(num_node_features, num_classes).to('cpu')

        test_mask = train(model, dataset, device,dataseed)

        acc= test(model, dataset, test_mask)
        accs.append(acc)
    print('Average accuracy: {:.4f}'.format(np.mean(accs)))
    print('Standard deviation: {:.4f}'.format(np.std(accs)))


def save_args_to_file(args, output_folder="configs"):
    filename = args.dataset+'_'+args.model_type+args.feature_type+'.config'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_file = os.path.join(output_folder, filename)

    # Convert args to string and save to file
    with open(output_file, 'w') as file:
        for arg in vars(args):
            file.write(f"{arg}: {getattr(args, arg)}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Graph Neural Network Training")
    parser.add_argument('--dataset', type=str, default = 'cora', help="Name of the dataset")
    parser.add_argument('--feature_type', type=str, default='ogb', help="Feature type for dataset")
    parser.add_argument('--device', type=str, default='cpu', help="Device to use for training")
    parser.add_argument('--model_type', type=str, default='GCN', help="Language model name")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of layers in GCN")
    parser.add_argument('--dim_hidden', type=int, default=64, help="Hidden dimension in GCN")
    parser.add_argument('--dropout', type=float, default=0.8, help="Dropout probability")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay for Adam optimizer")
    parser.add_argument('--transductive', type=bool, default=True, help="Transductive or Inductive setting")
    args = parser.parse_args()
    main(args)
    save_args_to_file(args)