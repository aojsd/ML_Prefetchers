import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from multiset_onehot import load_data_dir, print_results

def bit_split(X, splits, len_split, signed=True):
    # Separate splits in input based on bitwise values
    # Input X has shape (N, )
    # Output will have shape (N, 2*splits) if signed, else (N, splits)
    #   Lower order bits will have lower index in the splits dimension
    #   Output has a positive and negative section. If the original is positive,
    #       the negative section will be all zeroes, and vice versa
    X = torch.abs(X)
    mask = (1 << len_split) - 1
    out = torch.empty(X.numel(), 0, dtype=torch.long, device=X.device)
    for _ in range(splits):
        t_c = torch.bitwise_and(X, mask)
        out = torch.cat([out, t_c.unsqueeze(1)], dim=1)
        X >>= len_split
    if signed:
        # signs = torch.ge(X, 0).byte().unsqueeze(-1)
        # out = torch.cat([out * signs, out * (1-signs)], dim=1)
        out = torch.cat([out, out], dim=1)
    return out

def unsplit(X, splits, len_split, signed=True):
    # Convert splits back into integer values
    # Input X has shape:
    #   (N, splits) if unsigned
    #   (N, 1 + 2*splits) if signed
    # Output will have shape (N, )
    def get_int(B):
        coef = 1
        out = torch.zeros(B.shape[0], dtype=torch.long, device=B.device)
        for i in range(splits):
            out += coef * B[:, -i]
            coef <<= len_split
        return out
    
    if signed:
        # Defined such that positive values have sign=1, neg have sign=0
        # Sign bit at the beginning
        signs = X[:, 0]
        pos = get_int(X[:, 1:1+splits])
        neg = get_int(X[:, -splits:])
        return signs * pos - (1-signs) * neg
    else:
        return get_int(X)

# Module for weighting each bit/bit group value differently for the loss function
#   Default is to weight all equally
class MultibitSoftmax(nn.Module):
    def __init__(self, num_bits, splits, weights=None):
        super(MultibitSoftmax, self).__init__()
        # Assumes splits divides evenly with num_bits
        self.splits = splits
        self.len_split = int(num_bits/splits)
        self.CE = nn.CrossEntropyLoss()
        
        if weights == None:
            self.weights = torch.ones(1 << self.len_split)
        self.weight_CE = nn.CrossEntropyLoss(self.weights)

    def forward(self, X, target):
        # X holds inputs of shape (N, 2 * splits * (2^len_split))
        # target has shape (N, ), dtype = long
        # Output:
        #   preds --> shape (N, splits)
        #   loss ---> type float
        N, _ = X.shape

        # Reshape X and to separate splits for positive and negative cases
        ce_in = X.reshape(N, -1, 2*self.splits)
        # x_pos = ce_in[:, :, :self.splits]
        # x_neg = ce_in[:, :, self.splits:]

        # Separate splits in target based on bitwise values
        # ce_target will have shape (N, 2*splits)
        # Lower order bits will have lower index in the splits dimension
        ce_target = bit_split(target, self.splits, self.len_split)

        loss = self.CE(ce_in, ce_target)
        preds = ce_in.argmax(1)
        return preds, loss

    def predict(self, X):
        N = X.shape[0]
        x_splits = X.reshape(N, -1, 2*self.splits)
        preds = x_splits.argmax(1)
        return preds

class BitsplitEmbedding(nn.Module):
    def __init__(self, num_bits, splits, embedding_dim, signed=True):
        super(BitsplitEmbedding, self).__init__()

        # Assumes splits divides evenly into both num_bits and embedding_dim
        self.splits = splits
        self.len_split = int(num_bits/splits)
        self.signed = signed

        num_embedding = 1 << self.len_split
        if signed:
            num_embed = 2*splits
        else:
            num_embed = splits
        self.embeds = nn.ModuleList(
                        [nn.Embedding(num_embedding, embedding_dim)
                            for _ in range(num_embed)] )
    
    def forward(self, X):
        # X holds inputs of shape (N, )
        # Converts X to a tensor of shape (N, 2*splits) representing
        #   the bits in each split for positive and negative cases
        # Returns tensor of shape (N, 2*embedding_dim), if signed
        # else shape (N, embedding_dim)
        N = X.shape[0]

        # Separate splits in X based on bitwise values
        X = bit_split(X, self.splits, self.len_split, self.signed)

        # Perform multiple embeddings for each row in the batch
        out = torch.empty(N, 0, device=X.device)
        for i, E in enumerate(self.embeds):
            out = torch.cat([out, E(X[:,i])], dim=-1)
        return out

class SplitBinaryNet(nn.Module):
    def __init__(self, num_bits, embed_dim,  hidden_dim, num_layers=1,
                 dropout=0.1, splits=8, sign_weight=1):
        super(SplitBinaryNet, self).__init__()

        # Saved parameters
        self.num_bits = num_bits
        self.splits = splits
        self.len_split = int(num_bits/splits)
        self.num_classes = 1 << self.len_split
        self.sign_weight = sign_weight

        # Embedding layers
        self.addr_embed = BitsplitEmbedding(num_bits, splits, embed_dim, signed=False)
        s_dim = splits

        # Lstm layers
        if num_layers > 1:
            self.lstm = nn.LSTM(s_dim*embed_dim, hidden_dim, num_layers,
                                batch_first=True, dropout=dropout)
            self.lstm_drop = nn.Dropout(0)
        else:
            self.lstm = nn.LSTM(s_dim*embed_dim, hidden_dim, num_layers,
                                batch_first=True, dropout=0)
            self.lstm_drop = nn.Dropout(dropout)

        # Linear and output layers
        output_len = 2 * splits * self.num_classes
        self.lin_magnitude = nn.Linear(hidden_dim, output_len)
        self.lin_sign = nn.Linear(hidden_dim, 2)

        self.m_soft = MultibitSoftmax(num_bits, splits)
        self.CE = nn.CrossEntropyLoss()

    def forward(self, X, lstm_state, target):
        # X contains raw addresses, has shape (T,):
        # target is a tensor of the target deltas, has shape (T,)
        #       target deltas are not binarized
        # Returns loss, predictions, and lstm state
        embed_out = self.addr_embed(X)

        # Concatenate and feed into LSTM
        lstm_in = embed_out.unsqueeze(0)
        lstm_out, state = self.lstm(lstm_in, lstm_state)
        lstm_out = self.lstm_drop(lstm_out)

        # Linear Layers
        mag = self.lin_magnitude(lstm_out).squeeze(1)
        sign_probs = self.lin_sign(lstm_out).squeeze(1)

        # Loss and prediction calculation
        mag_preds, mag_loss = self.m_soft(mag, target)
        sign_preds = sign_probs.argmax(-1).unsqueeze(-1)
        target_signs = torch.ge(target, 0).long()
        sign_loss = self.CE(sign_probs, target_signs)

        # Final weighted loss and predictions
        loss = mag_loss + self.sign_weight * sign_loss
        preds = torch.cat([sign_preds, mag_preds], dim=-1)
        return loss, preds, state

    def predict(self, X, lstm_state):
        embed_out = self.addr_embed(X)
        lstm_in = embed_out.unsqueeze(0)
        lstm_out, state = self.lstm(lstm_in, lstm_state)
        mag = self.lin_magnitude(lstm_out.squeeze(dim=0))
        sign_probs = self.lin_sign(lstm_out).squeeze(dim=0)

        mag_preds = self.m_soft.predict(mag)
        sign_preds = sign_probs.argmax(-1).unsqueeze(-1)
        preds = torch.cat([sign_preds, mag_preds], dim=-1)
        return preds, state

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
            out, state = net.predict(X, state)

            # Detach to save memory
            out = out.detach()
            state = tuple([s.detach() for s in list(state)])

            # Calculate accuracy
            res = unsplit(out, net.splits, net.len_split) == target
            acc = res.sum() / res.numel()
            val_acc_list.append(acc)

        # Calculate overall accuracy for single dataset
        val_acc = torch.tensor(val_acc_list).mean()
        iter_acc_list.append(val_acc)

    return iter_acc_list

# def SplitBinary_eval(net, data_iter, device='cpu', state=None):
#     # Compute validation accuracy
#     net.eval()
#     val_acc_list = []
#     for _, data in enumerate(data_iter):
#         X = data[0].to(device)
#         target = data[1].to(device)
#         preds, state = net.predict(X, state)

#         # Detach to save memory
#         preds = preds.detach()
#         state = tuple([s.detach() for s in list(state)])

#         # Calculate accuracy
#         res = unsplit(preds, net.splits, net.len_split) == target
#         acc = res.sum() / res.numel()
#         val_acc_list.append(acc)

#     # Calculate overall accuracy
#     val_acc = torch.tensor(val_acc_list).mean()

#     return val_acc, state

def multiset_train(net, data_iters, iter_sizes, epochs, optimizer, device='cpu',
                   scheduler=None, print_interval=10, e_start=0):
    # Calculate relative probability to replay for differently sized datasets
    min_dsize = min(iter_sizes)
    iter_probs = [x / min_dsize for x in iter_sizes]

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
                out, state = net.predict(X, state)

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
            # out, state = net(X, state)
            optimizer.zero_grad()
            loss, out, state = net(X, state, target)
            # loss = F.cross_entropy(out, target, ignore_index=max_classes)

            loss.backward()
            epoch_loss.append(loss.detach())
            optimizer.step()

        
            # Detach state gradients to avoid autograd errors
            state = tuple([s.detach() for s in list(state)])

            # Update replay list with new state, except when at start of new sequence
            # replay_state = (state[0].to('cpu'), state[1].to('cpu'))
            # if i+1 < index and i+1 not in seq_starts:
            #     x, t, st = replay_list[i+1]
            #     replay_list[i+1] = x, t, replay_state

            # Calculate accuracy
            preds = unsplit(out, net.splits, net.len_split)
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
    addr_in = torch.tensor(data[0].to_numpy())
    targets = torch.tensor(data[1].to_numpy())
    return addr_in, targets

# Setup data iterators
def setup_data(features, targets, batch_size=2):
    dataset = torch.utils.data.TensorDataset(
                    features, targets)
    data_iter = torch.utils.data.DataLoader(
                    dataset, batch_size, shuffle=False)
    return data_iter

def main(args):
    # Reproducibility
    torch.manual_seed(0)

    # Retrive data from file
    datadir = args.datadir
    batch_size = args.batch_size
    train_size = args.train_size
    data_iters, iter_sizes, _, _, _ = load_data_dir(datadir, None, args.prefix, batch_size, train_size)

    # Model parameters
    num_bits = 36
    splits = 12
    e_dim = args.eh
    h_dim = args.eh
    layers = 1
    dropout = 0.1

    # Create net
    net = SplitBinaryNet(num_bits, e_dim, h_dim, layers, splits=splits, dropout=dropout)
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
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25, gamma=0.32)

    # Train model
    tup = multiset_train(net, data_iters, iter_sizes, epochs, optimizer, device=device, scheduler=scheduler,
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
    dataset_accs = multiset_eval(net.to(device), data_iters, device=device)
    print_results(datadir, args.prefix, dataset_accs)

    # # Validation test
    # if not args.t:
    #     val_iter = setup_data(addr_in, target, batch_size=batch_size)
    #     val_acc, state = SplitBinary_eval(net, val_iter, device=device)
    #     print(f"Val Accuracy:\t{val_acc:.6f}")

    # # Test model
    # if args.t:
    #     # Use 75/25 for val/test
    #     h = int(len(addr_in) * 1/2)
    #     val_addr = addr_in[:h]
    #     val_targ = target[:h]
    #     val_iter = setup_data(val_addr, val_targ, batch_size=batch_size)

    #     # Final training on validation
    #     optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    #     scheduler = None
    #     tup = SplitBinary_train(net, val_iter, 1, optimizer, device=device, scheduler=scheduler,
    #                             print_interval=5, e_start = 0)
    #     val_acc, state = SplitBinary_eval(net, val_iter, device=device)
    #     print(f"Val Accuracy:\t{val_acc:.6f}")

    #     test_addr = addr_in[h:]
    #     test_targ = target[h:]
    #     test_iter = setup_data(test_addr, test_targ, batch_size=batch_size)
    #     test_acc, _ = SplitBinary_eval(net, test_iter, device=device, state=state)
    #     print(f"Test Accuracy:\t{test_acc:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datadir", help="Directory containing input datasets to train/test on", type=str)
    parser.add_argument("--prefix", help="Shared prefix to search for files in directory", type=str, default="")
    parser.add_argument("--train_size", help="Size of training set", default=-1, type=int)
    parser.add_argument("--batch_size", help="Batch size for training", default=64, type=int)
    parser.add_argument("--epochs", help="Number of epochs to train", default=10, type=int)
    parser.add_argument("--init_epochs", help="Number of epochs already pretrained", default=0, type=int)
    parser.add_argument("--print", help="Print loss during training", default=1, type=int)
    parser.add_argument("--nocuda", help="Don't use cuda", action="store_true", default=False)
    parser.add_argument("--model_file", help="File to load/save model parameters to continue training", default=None, type=str)
    parser.add_argument("--trend_file", help="File to save trends and results", default=None, type=str)
    parser.add_argument("--lr", help="Initial learning rate", default=1e-5, type=float)
    parser.add_argument("--eh", help="Embedding and hidden dimensions", type=int, default=32)

    args = parser.parse_args()
    main(args)