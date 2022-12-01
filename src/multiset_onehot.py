import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from one_hot_pages_model import OneHotNet

def multiset_train(net, data_iters, iter_sizes, epochs, optimizer, replay_prob=1, device='cpu',
                   scheduler=None, print_interval=10, e_start=0, max_classes=20000):
    # Calculate relative probability to replay for differently sized datasets
    min_dsize = min(iter_sizes)
    iter_probs = [replay_prob * x / min_dsize for x in iter_sizes]

    # Begin training
    loss_list = []
    acc_list = []
    net.train()
    for e in range(epochs):
        epoch_loss = []
        epoch_acc  = []

        # Get replay list after one pass-through for interleaving
        replay_list = []
        seq_starts = []
        index = 0
        for d_iter, rp, sz in zip(data_iters, iter_probs, iter_sizes):
            # Batch randomization
            replays = (torch.rand(sz) < rp).tolist()

            state = None
            for data, to_replay in zip(d_iter, replays):
                # Add example and initial state to replay list
                if to_replay:
                    if state != None:
                        replay_state = (state[0].to('cpu'), state[1].to('cpu'))
                    else:
                        replay_state = None
                    replay_list.append((data[0], data[1], replay_state))
                    index += 1

                # Forward pass
                X = data[0].to(device)
                out, state = net(X, state)

                # Detach state gradients to avoid autograd errors
                state = tuple([s.detach() for s in list(state)])

            seq_starts.append(index)

        # Shuffle replay list for interleaved learning
        epoch_list = torch.randperm(index).tolist()
        for i in epoch_list:
            x, targ, st = replay_list[i]

            X = x.to(device)
            target = targ.to(device)
            if st != None:
                state = (st[0].to(device), st[1].to(device))
            else:
                state = None
            out, state = net(X, state)

            optimizer.zero_grad()
            loss = F.cross_entropy(out, target, ignore_index=max_classes)

            # NAN check
            if torch.isnan(loss).sum() == 0:
                loss.backward()
                epoch_loss.append(loss.detach())
            optimizer.step()

        
            # Detach state gradients to avoid autograd errors
            state = tuple([s.detach() for s in list(state)])

            # Update replay list with new state, except when at start of new sequence
            replay_state = (state[0].to('cpu'), state[1].to('cpu'))
            if i+1 < index and i+1 not in seq_starts:
                x, t, st = replay_list[i+1]
                replay_list[i+1] = x, t, replay_state

            # Calculate accuracy
            preds = out.argmax(dim=-1)
            res = preds == target
            acc = res.sum() / res.numel()
            epoch_acc.append(acc.detach())

        if scheduler != None:
            scheduler.step()
        
        loss = torch.Tensor(epoch_loss).mean().item()
        acc = torch.Tensor(epoch_acc).mean().item()
        loss_list.append(loss)
        acc_list.append(acc)
        if (e+1) % print_interval == 0:
            print(f"Epoch {e+1 + e_start}\tLoss:\t{loss:.8f}\tTrain Accuracy:\t{acc:.4f}")

    return loss_list, acc_list

def multiset_eval(net, data_iters, device='cpu'):
    # Compute validation accuracy
    net.eval()
    iter_acc_list = []
    for d_iter in data_iters:
        val_acc_list = []
        state = None
        for data in d_iter:
            X = data[0].to(device)
            target = data[1].to(device)
            out, state = net(X, state)

            # Detach to save memory
            out = out.detach()
            state = tuple([s.detach() for s in list(state)])

            # Calculate accuracy
            res = out.argmax(dim=-1) == target
            acc = res.sum() / res.numel()
            val_acc_list.append(acc)

        # Calculate overall accuracy for single dataset
        val_acc = torch.tensor(val_acc_list).mean()
        iter_acc_list.append(val_acc)

    return iter_acc_list

# Merge data from all csv files in a directory
def load_data_dir(dir, max_classes, prefix="", batch_size=128, nrows=-1):
    # Things to return
    data_iters = []
    iter_sizes = []
    index_map = {}
    rev_map = {}

    # Read in all file data
    d_frames = []
    freqs = []
    csvs = []
    temp_index = {}
    temp_rev = {}
    count = 0
    for f in os.listdir(dir):
        if f.startswith(prefix) and f.endswith(".csv"):
            fname = dir + "/" + f
            if nrows != -1:
                data = pd.read_csv(fname, nrows=nrows, header=None)
            else:
                data = pd.read_csv(fname, header=None)
            csvs.append(data)
            iter_sizes.append(len(data.index))

    min_sz = min(iter_sizes)
    for data, sz in zip(csvs, iter_sizes):
        # Get unique values and their frequencies, weighted by sequence length
        if max_classes != None:
            f_weight = min_sz/sz
            for d in data[0]:
                if d not in temp_index:
                    temp_index[d] = count
                    temp_rev[count] = d
                    freqs.append(f_weight)
                    count += 1
                freqs[temp_index[d]] += f_weight

        d_frames.append(data)

    # Produce index and reverse maps for only the most frequent values
    if max_classes != None:
        if len(freqs) > max_classes:
            count = 0
            top_indices = torch.topk(torch.Tensor(freqs), max_classes)[1].tolist()
            for i in top_indices:
                v = temp_rev[i]
                index_map[v] = count
                rev_map[count] = v
                count += 1
            
            del temp_index
            del temp_rev
        else:
            max_classes = count
            index_map = temp_index
            rev_map = temp_rev

        # Function to replace data with indices instead of raw values, infrequent values set to invalid index
        def index_replace(x):
            if x in index_map:
                return index_map[x]
            else:
                return max_classes

    # Setup data loaders from dataframes
    for data in d_frames:
        if max_classes != None:
            data = data.applymap(index_replace)
        addr_in = torch.tensor(data[0].to_numpy())
        targets = torch.tensor(data[1].to_numpy())

        d_set  = torch.utils.data.TensorDataset(addr_in, targets)
        d_iter = torch.utils.data.DataLoader(d_set, batch_size, shuffle=False)
        data_iters.append(d_iter)

        del addr_in
        del targets
        del data
    del d_frames

    return data_iters, iter_sizes, index_map, rev_map, max_classes

# Print validation accuracy results for each input file
def print_results(dir, prefix, val_accs):
    files = [f for f in os.listdir(dir) if f.startswith(prefix)]
    for fname, acc in zip(files, val_accs):
        print(f"Accuracy for {fname}:\t{acc:.6f}")

def main(args):
    # Reproducibility
    torch.manual_seed(0)

    # Retrive data from file
    datadir = args.datadir
    max_classes = args.max_classes
    batch_size = args.batch_size
    train_size = args.train_size
    data_iters, iter_sizes, _, _, num_classes = load_data_dir(datadir, max_classes, args.prefix, batch_size, train_size)
    print("Number of Classes: " + str(num_classes))

    # Model parameters
    e_dim = args.eh
    h_dim = args.eh
    layers = 1
    dropout = 0.1

    # Create net
    net = OneHotNet(max_classes, e_dim, h_dim, layers, dropout=dropout)
    if not args.nocuda:
        device = torch.device('cuda:0')
        net = net.to(device)
    else:
        device = 'cpu'

    # Check for model file
    if args.model_file != None:
        if os.path.exists(args.model_file):
            print("Loading model from file: {}".format(args.model_file))
            net.load_state_dict(torch.load(args.model_file))
        else:
            print("Creating model file: {}".format(args.model_file))

    # Training parameters
    epochs = args.epochs
    print_in = args.print
    e_start = args.init_epochs
    lr = args.lr
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = None

    # Train model
    tup = multiset_train(net, data_iters, iter_sizes, epochs, optimizer, device=device, scheduler=scheduler,
                            print_interval=print_in, e_start = e_start, max_classes=max_classes)
    train_loss_list, train_acc_list = tup

    # Save model parameters
    if args.model_file != None:
        torch.save(net.cpu().state_dict(), args.model_file)

    # Save training trends
    if args.trend_file != None:
        trends = pd.DataFrame(zip(train_loss_list, train_acc_list), columns=['loss', 'val'])
        trends.to_csv(args.trend_file, index=False)


    # Validation test
    dataset_accs = multiset_eval(net.to(device), data_iters, device=device)
    print_results(datadir, args.prefix, dataset_accs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datadir", help="Directory containing input datasets to train/test on", type=str)
    parser.add_argument("--prefix", help="Shared prefix to search for files in directory", type=str, default="")
    parser.add_argument("--train_size", help="Size of training set", default=-1, type=int)
    parser.add_argument("--batch_size", help="Batch size for training", default=512, type=int)
    parser.add_argument("--epochs", help="Number of epochs to train", default=10, type=int)
    parser.add_argument("--init_epochs", help="Number of epochs already pretrained", default=0, type=int)
    parser.add_argument("--print", help="Print loss during training", default=1, type=int)
    parser.add_argument("--nocuda", help="Don't use cuda", action="store_true", default=False)
    parser.add_argument("--model_file", help="File to load/save model parameters to continue training", default=None, type=str)
    parser.add_argument("--trend_file", help="File to save trends and results", default=None, type=str)
    parser.add_argument("--lr", help="Initial learning rate", default=1e-3, type=float)
    parser.add_argument("--max_classes", help="Max classes for addresses/deltas", type=int, default=20000)
    parser.add_argument("--eh", help="Embedding and hidden dimensions", type=int, default=32)

    args = parser.parse_args()
    main(args)