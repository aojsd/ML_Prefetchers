import argparse
import os
from typing import Optional, Tuple
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

class OneHotNet(nn.Module):
    def __init__(self, num_classes, embed_dim,  hidden_dim, num_layers=1, dropout=0.1):
        super(OneHotNet, self).__init__()
        # Embedding layers
        self.embed = nn.Embedding(num_classes+1, embed_dim)
        self.h_dim = hidden_dim

        # Lstm layers
        if num_layers > 1:
            self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
            self.lstm_drop = nn.Dropout(0)
        else:
            self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=0)
            self.lstm_drop = nn.Dropout(dropout)

        # Linear and output layers
        self.lin = nn.Linear(hidden_dim, num_classes+1)

    def forward(self, X, lstm_state : Optional[Tuple[torch.Tensor, torch.Tensor]], num_preds=1):
        # X contains raw addresses, has shape (T,):
        # target is a tensor of the target delta probabilities, has shape (T, num_classes)
        # Returns predictions and lstm state
        embed_out = self.embed(X)

        # Concatenate and feed into LSTM
        lstm_in = embed_out.unsqueeze(0)
        lstm_out, state = self.lstm(lstm_in, lstm_state)
        lstm_out = self.lstm_drop(lstm_out)

        # Linear Layer
        out = self.lin(F.relu(lstm_out)).squeeze(0)

        # Multiple predictions
        while num_preds > 1:
            num_preds -= 1
            next_in = out.argmax(-1)
            embed_out = self.embed(next_in)
            lstm_in = embed_out.unsqueeze(0)
            lstm_out, state = self.lstm(lstm_in, lstm_state)
            lstm_out = self.lstm_drop(lstm_out)
            out = self.lin(F.relu(lstm_out)).squeeze(0)

        return out, state

def OHNet_eval(net, data_iter, device='cpu', state=None):
    # Compute validation accuracy
    net.eval()
    val_acc_list = []
    for _, data in enumerate(data_iter):
        X = data[0].to(device)
        target = data[1].to(device)
        out, state = net(X, state, 1)

        # Detach to save memory
        out = out.detach()
        state = tuple([s.detach() for s in list(state)])

        # Calculate accuracy
        res = out.argmax(dim=-1) == target
        acc = res.sum() / res.numel()
        val_acc_list.append(acc)

    # Calculate overall accuracy
    val_acc = torch.tensor(val_acc_list).mean()

    return val_acc, state

def OHNet_train(net, data_iter, epochs, optimizer, device='cpu', scheduler=None,
                      print_interval=10, e_start=0):
    loss_list = []
    acc_list = []

    net.train()
    for e in range(epochs):
        epoch_loss = []
        epoch_acc  = []

        # Get replay list after one pass-through for interleaving
        replay_list = []
        index = 0
        state = None
        # for _, data in enumerate(data_iter):
        for data in data_iter:
            # Add example and initial state to replay list
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

        # Shuffle replay list for interleaved learning
        epoch_list = torch.randperm(index)
        for j in epoch_list:
            i = j.item()
            x, targ, st = replay_list[i]

            X = x.to(device)
            target = targ.to(device)
            if st != None:
                state = (st[0].to(device), st[1].to(device))
            else:
                state = None
            out, state = net(X, state)

            loss = F.cross_entropy(out, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.detach())
        
            # Detach state gradients to avoid autograd errors
            state = tuple([s.detach() for s in list(state)])

            # Update replay list with new state
            replay_state = (state[0].to('cpu'), state[1].to('cpu'))
            if i+1 < index:
                x, t, _ = replay_list[i+1]
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

# Load data from csv file
def load_data(infile, nrows=-1):
    if nrows != -1:
        data = pd.read_csv(infile, nrows=nrows, header=None)
    else:
        data = pd.read_csv(infile, header=None)

    index_map = {}
    rev_map = {}
    count = 0
    for d in data[0]:
        if d not in index_map:
            index_map[d] = count
            rev_map[count] = d
            count += 1
    if data.iloc[-1,-1] not in index_map:
        index_map[data.iloc[-1,-1]] = count
        rev_map[count] = data.iloc[-1,-1]
        count += 1
    data = data.applymap(lambda x: index_map[x])

    addr_in = torch.tensor(data[0].to_numpy())
    targets = torch.tensor(data[1].to_numpy())

    del data
    return addr_in, targets, index_map, rev_map, count

# Setup data iterators
def setup_data(features, targets, batch_size=2):
    dataset = torch.utils.data.TensorDataset(
                    features, targets)
    data_iter = torch.utils.data.DataLoader(
                    dataset, batch_size, shuffle=False)
    del features
    del targets
    return data_iter

def main(args):
    # Reproducibility
    torch.manual_seed(0)

    # Retrive data from file
    datafile = args.datafile
    train_size = args.train_size
    batch_size = args.batch_size
    if train_size == -1:
        addr_in, target, _, _, num_classes = load_data(datafile)
    else:
        addr_in, target, _, _, num_classes = load_data(datafile, train_size)
    print("Number of Classes: " + str(num_classes))

    # Model parameters
    e_dim = 256
    h_dim = 256
    layers = 1
    dropout = 0.1

    # Create net
    net = OneHotNet(num_classes, e_dim, h_dim, layers, dropout=dropout)
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
            if not args.t:
                print("Creating model file: {}".format(args.model_file))
            else:
                print("No model given to test (model file does not exist)")
                exit()

    if not args.t:
        # Setup data
        data_iter = setup_data(addr_in, target, batch_size=batch_size)

        # Training parameters
        epochs = args.epochs
        print_in = args.print
        e_start = args.init_epochs
        lr = args.lr
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        scheduler = None
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25, gamma=0.32)

        # Train model
        tup = OHNet_train(net, data_iter, epochs, optimizer, device=device, scheduler=scheduler,
                                print_interval=print_in, e_start = e_start)
        train_loss_list, train_acc_list = tup

        # Save model parameters
        if args.model_file != None:
            torch.save(net.cpu().state_dict(), args.model_file)

        # Save training trends
        trends = pd.DataFrame(zip(train_loss_list, train_acc_list), columns=['loss', 'val'])
        if args.trend_file != None:
            trends.to_csv(args.trend_file, index=False)


        # Validation test
        val_acc, state = OHNet_eval(net.to(device), data_iter, device=device)
        print(f"Val Accuracy:\t{val_acc:.6f}")

    # Test model
    if args.t:
        # Use 75/25 for val/test
        h = int(len(addr_in) * 1/2)
        val_addr = addr_in[:h]
        val_targ = target[:h]
        val_iter = setup_data(val_addr, val_targ, batch_size=batch_size)

        # Final training on validation
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        scheduler = None
        tup = OHNet_train(net, val_iter, 1, optimizer, device=device, scheduler=scheduler,
                                print_interval=5, e_start = 0)
        val_acc, state = OHNet_eval(net, val_iter, device=device)
        print(f"Val Accuracy:\t{val_acc:.6f}")

        test_addr = addr_in[h:]
        test_targ = target[h:]
        test_iter = setup_data(test_addr, test_targ, batch_size=batch_size)
        test_acc, _ = OHNet_eval(net, test_iter, device=device, state=state)
        print(f"Test Accuracy:\t{test_acc:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile", help="Input data set to train/test on", type=str)
    parser.add_argument("--train_size", help="Size of training set", default=-1, type=int)
    parser.add_argument("--batch_size", help="Batch size for training", default=256, type=int)
    parser.add_argument("--epochs", help="Number of epochs to train", default=10, type=int)
    parser.add_argument("--init_epochs", help="Number of epochs already pretrained", default=0, type=int)
    parser.add_argument("--print", help="Print loss during training", default=1, type=int)
    parser.add_argument("--nocuda", help="Don't use cuda", action="store_true", default=False)
    parser.add_argument("--model_file", help="File to load/save model parameters to continue training", default=None, type=str)
    parser.add_argument("--trend_file", help="File to save trends and results", default=None, type=str)
    parser.add_argument("--lr", help="Initial learning rate", default=1e-3, type=float)
    parser.add_argument("-t", help="Evaluate on test set", action="store_true", default=False)

    args = parser.parse_args()
    main(args)