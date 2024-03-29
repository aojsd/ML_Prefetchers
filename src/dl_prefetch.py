import random
import numpy as np
import torch
import torch.nn.functional as F
from group_prefetcher import *
from typing import Optional, Tuple

# Replay memories
class ReplayMemory():
    def __init__(self, hard=[], med=[], easy=[]):
        self.hard = hard
        self.med = med
        self.easy = easy

    def sample(self, h=0, m=0, e=0):
        if h <= len(self.hard):
            h_samp = random.sample(list(range(len(self.hard))), h)
        else:
            h_samp = []

        if m <= len(self.med):
            m_samp = random.sample(list(range(len(self.med))), m)
        else:
            m_samp = []

        if e <= len(self.easy):
            e_samp = random.sample(list(range(len(self.easy))), e)
        else:
            e_samp = []
        
        return h_samp, m_samp, e_samp


# Prefetcher using a deep neural network
class DLPrefetcher():
    def __init__(self, model, device, oh_params=None, online_params=None, lr=1e-4, only1=False, window=None):
        self.model = model.to(device)
        self.device = device
        self.state = None
        self.oh = oh_params
        self.only1 = only1
        self.win = window

        if oh_params != None:
            if oh_params.class_limit == None:
                self.inv_class = len(oh_params.imap)
                self.c_limit = False
            else:
                self.inv_class = oh_params.class_limit
                self.c_limit = True
                self.c_alloced = 0
                

        self.online = False
        self.online_params = online_params
        self.init_replay = False
        if online_params != None:
            self.online = True
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

            self.replay_mem = None
            self.other_mem = None
            if online_params.replay:
                self.replay_mem = ReplayMemory()
                self.other_mem = ReplayMemory()
                self.loss_vec = None
                self.count = 0
        self.x = None

        if self.win != None:
            self.hist = [0 for _ in range(self.win)]
            self.h_tl = 0
            self.filled = False

            def loss_fn(x, y):
                loss = 0
                x = torch.sigmoid(x)
                return F.multilabel_margin_loss(x, y)
            # weight = torch.ones(self.inv_class+1)
            # weight[self.inv_class] = 0
            # self.loss_fn = torch.nn.MultiLabelMarginLoss()
            self.loss_fn = torch.nn.MultiLabelSoftMarginLoss()
            # self.loss_fn = loss_fn
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.inv_class)

    def learn(self, x):
        if self.win != None:
            self.hist[self.h_tl] = x
            self.h_tl = (self.h_tl + 1) % self.win
            self.filled = self.filled or self.h_tl == 0
            if not self.filled:
                return
            # For raw addresses
            # window = [a - self.hist[self.h_tl] for a in self.hist]

            # For deltas
            diff = 0
            window = []
            for i in range(self.win):
                index = (self.h_tl + i) % self.win
                window.append(self.hist[index] + diff)
                diff += self.hist[index]

        # For onehot model, convert addresses to indices
        def map(x, add=True):
            if x in self.oh.imap:
                x = self.oh.imap[x]
            elif add and self.c_limit and self.c_alloced < self.inv_class:
                self.oh.imap[x] = self.c_alloced
                self.oh.rmap[self.c_alloced] = x

                x = self.c_alloced
                self.c_alloced += 1
            else:
                x = self.inv_class
            return x
        
        if self.oh != None:
            if self.win != None:
                x = [map(a) for a in window]
                x = torch.zeros(self.inv_class+1).scatter_(0, torch.tensor(x), 1)
            else:
                x = torch.tensor([map(x)])

        # Function for weight update
        #   Both args should be tensors
        def do_update(input, output, st):
            if st != None:
                state = tuple(s.to(self.device) for s in st)
            else:
                state = None
            self.optimizer.zero_grad()

            # Forward pass
            out, _ = self.model(input.to(self.device), state)
            if len(out.shape) > 2:
                out = out.squeeze()

            # Backward pass
            loss = self.loss_fn(out, output.to(self.device))
            loss.backward()
            self.optimizer.step()
            return loss.detach().cpu().item()

        # Online learning
        if self.online and not self.x == None and not (self.win == None and x == self.inv_class):
            if not self.init_replay:
                self.model.train()
                loss = do_update(self.x, x, self.state)

            # Interleaved learning using replay memory
            if self.online_params.replay:
                if self.count % self.online_params.rep_fr == 0 and not self.init_replay:
                    # Function to iterate through list of examples to replay
                    def iterate_replay(rlist, rmem):
                        inputs = []
                        targs = []
                        h = []
                        c = []
                        for i in rlist:
                            x_in, targ, state = rmem[i]
                            inputs.append(x_in.unsqueeze(0))
                            targs.append(targ)
                            h.append(state[0])
                            c.append(state[1])

                        inputs = torch.cat(inputs, dim=0)
                        targs = torch.cat(targs, dim=0)
                        h = torch.cat(h, dim=1)
                        c = torch.cat(c, dim=1)

                        do_update(inputs, targs, (h,c))

                    # Sample replay examples
                    tot = self.online_params.rep_len
                    hs, ms, es = self.replay_mem.sample(0, 0, tot)

                    # Do replay learning                    
                    iterate_replay(es, self.replay_mem.easy)

                # Add to replay memory with some probability
                sample = (self.x, torch.tensor([x]), tuple(s.cpu() for s in self.state))
                if self.online_params.add:
                    if self.online_params.clear and not self.init_replay:
                        self.other_mem.easy.append(sample)
                    self.replay_mem.easy.append(sample)

        if self.win != None:
            x = torch.tensor([map(self.hist[self.h_tl], False)])
        self.x = x

    def predict(self, k=1):
        if self.online:
            self.model.eval()
        if self.oh != None:
            if self.win != None and not self.filled:
                return [(0, 0)]
            model_out, state = self.model(self.x.to(self.device), self.state)
            self.state = tuple(s.detach() for s in state)
            
            # Reverse map indices to addresses
            def ind(x, c):
                if x in self.oh.rmap:
                    p = self.oh.rmap[x]
                    if self.only1:
                        if p == 1:
                            return 1, c
                        else: return 1, 0
                    else:
                        return p, c
                else:
                    return None, 0
            if k == 1:
                item = model_out.squeeze().topk(1)
                preds = [ind(item[1].item(), item[0].item())]
                return preds
            else:
                out = model_out.squeeze().topk(k)
                preds = [ind(b.item(), a.item()) for a, b in zip(out[0], out[1])]
                return preds
        else:
            model_out, self.state = self.model.predict(self.x, self.state)
            out = unsplit(model_out, self.model.splits, self.model.len_split)
            return [(out.item(), 1)]
        
    def print_parameters(self):
        return

class SplitBinaryPref():
    def __init__(self, model, device, online_params=None, lr=1e-3):
        self.model = model.to(device)
        self.device = device
        self.state = None

        self.online = False
        self.online_params = online_params
        self.init_replay = False
        if online_params != None:
            self.online = True
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

            self.replay_mem = None
            self.other_mem = None
            if online_params.replay:
                self.replay_mem = ReplayMemory()
                self.other_mem = ReplayMemory()
                self.loss_vec = None
                self.count = 0
        self.x = None

    def learn(self, x):
        # Function for weight update
        #   Both args should be tensors
        def do_update(input, output, st):
            if st != None:
                state = tuple(s.to(self.device) for s in st)
            else:
                state = None
            self.model.train()
            self.optimizer.zero_grad()
            loss, _, _ = self.model(input.to(self.device), state, output.to(self.device))
            loss.backward()
            self.optimizer.step()
            return loss.detach().cpu().item()

        # Online learning
        if self.online and not self.x == None:
            if not self.init_replay:
                loss = do_update(self.x, torch.tensor([x]), self.state)

            # Interleaved learning using replay memory
            # if self.online_params.replay:
            #     if self.count % self.online_params.rep_fr == 0 and not self.init_replay:
            #         # Function to iterate through list of examples to replay
            #         def iterate_replay(rlist, rmem):
            #             inputs = []
            #             targs = []
            #             h = []
            #             c = []
            #             for i in rlist:
            #                 x_in, targ, state = rmem[i]
            #                 inputs.append(x_in.unsqueeze(0))
            #                 targs.append(targ)
            #                 h.append(state[0])
            #                 c.append(state[1])

            #             inputs = torch.cat(inputs, dim=0)
            #             targs = torch.cat(targs, dim=0)
            #             h = torch.cat(h, dim=1)
            #             c = torch.cat(c, dim=1)

            #             do_update(inputs, targs, (h,c))

            #         # Sample replay examples
            #         tot = self.online_params.rep_len
            #         hs, ms, es = self.replay_mem.sample(0, 0, tot)

            #         # Do replay learning                    
            #         iterate_replay(es, self.replay_mem.easy)

            #     # Add to replay memory with some probability
            #     sample = (self.x, torch.tensor([x]), tuple(s.cpu() for s in self.state))
            #     if self.online_params.add:
            #         if self.online_params.clear and not self.init_replay:
            #             self.other_mem.easy.append(sample)
            #         self.replay_mem.easy.append(sample)

        self.x = torch.tensor([x])

    def predict(self, k=1):
        if self.online:
            self.model.eval()
        model_out, state = self.model.predict(self.x.to(self.device), self.state)
        self.state = tuple(s.detach() for s in state)
        out = unsplit(model_out, self.model.splits, self.model.len_split)
        return [(out.item(), 1)]
        
    def print_parameters(self):
        return
    
    

class Voyager(nn.Module):
    def __init__(self, num_classes, embed_dim, hidden_dim, num_layers=1, dropout=0.1, n_deltas=16):
        super(Voyager, self).__init__()
        # Embedding layers
        self.n_d = n_deltas
        self.c = num_classes + 2*n_deltas + 1
        self.embed = nn.Embedding(self.c, embed_dim)
        self.h_dim = hidden_dim

        # Lstm layers
        if num_layers > 1:
            self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
            self.lstm_drop = nn.Dropout(0)
        else:
            self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=0)
            self.lstm_drop = nn.Dropout(dropout)

        # Linear and output layers
        self.lin = nn.Linear(hidden_dim, self.c)

    def forward(self, X):
        # X contains raw addresses, has shape (N, L):
        # target is a tensor of the target delta probabilities, has shape (N, num_classes)
        # Returns predictions and lstm state
        embed_out = self.embed(X)

        # Concatenate and feed into LSTM
        if len(embed_out.shape) < 3:
            lstm_in = embed_out.unsqueeze(0)
        else:
            lstm_in = embed_out
        _, (lstm_out, _) = self.lstm(lstm_in, None)
        lstm_out = self.lstm_drop(lstm_out)

        # Linear Layer
        out = self.lin(F.relu(lstm_out)).squeeze(0)

        return out
    
class V_prefetcher():
    def __init__(self, model, device, window, oh_params, online_params=None, lr=1e-4, n_deltas=64):
        self.model = model.to(device)
        self.device = device
        self.state = None
        self.oh = oh_params
        self.win = window
        self.n_d = n_deltas

        if oh_params.class_limit == None:
            self.inv_class = len(oh_params.imap)
            self.c_limit = False
        else:
            self.inv_class = oh_params.class_limit
            self.c_limit = True
            self.c_alloced = 0
        self.inv_class += 2*n_deltas

        self.online = False
        self.online_params = online_params
        if online_params != None:
            self.online = True
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.x = None

        self.hist = [None for _ in range(self.win)]
        self.h_tl = 0
        self.filled = False

        # self.loss_fn = torch.nn.MultiLabelMarginLoss()
        # self.loss_fn = torch.nn.MultiLabelSoftMarginLoss()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
        # Statistics
        self.addr_preds = 0
        self.delta_preds = 0
    
    # Convert address or delta to class
    def map(self, addr, add=True):
        if abs(addr) <= self.n_d:
            if addr > 0:
                addr -= (self.n_d + 1)
            elif addr < 0:
                addr -= self.n_d
            else:
                print("Error: received delta of 0")
                exit()
            idx = self.inv_class + addr
        elif addr in self.oh.imap:
            idx = self.oh.imap[addr]
        elif add and self.c_limit and self.c_alloced < self.oh.class_limit:
            self.oh.imap[addr] = self.c_alloced
            self.oh.rmap[self.c_alloced] = addr

            idx = self.c_alloced
            self.c_alloced += 1
        else:
            idx = self.inv_class
        return idx
    
    # Convert history of addresses to classes and deltas
    def hist_idx(self, trans_only=False):
        base = self.hist[self.h_tl]
        extract = lambda i, y: base if y == base else (y - self.hist[(i-1) % self.win] if abs(y - self.hist[(i-1) % self.win]) <= self.n_d else y)
        transformed = [extract(i, y) for i, y in enumerate(self.hist)]
        
        if trans_only:
            return transformed
        return [self.map(y) for y in transformed]


    def learn(self, x):
        # Update history window only if not full
        if not self.filled:
            self.hist[self.h_tl] = x
            self.h_tl = (self.h_tl + 1) % self.win
            self.filled = self.filled or self.h_tl == 0
            return

        # Create input window
        # print(self.hist)
        hist_in = self.hist_idx()
        
        # Create indices tensor and rotate to correctly-ordered history
        hist_in = torch.tensor(np.roll(hist_in, -self.h_tl))
        
        # Setup target value
        prev_addr = self.hist[(self.h_tl - 1) % self.win]
        delta = x - prev_addr
        if delta == 0:
            return
        targ = delta if abs(delta) <= self.n_d else x
        targ = self.map(targ)

        # Function for weight update
        #   Both args should be tensors
        def do_update(input, targ):
            self.optimizer.zero_grad()

            # Forward pass
            out = self.model(input.to(self.device)).squeeze()
            # if len(out.shape) > 2:
            #     out = out.squeeze()

            # Backward pass
            targ = torch.tensor([targ]).to(self.device)
            loss = self.loss_fn(out.unsqueeze(0), targ)
            loss.backward()
            self.optimizer.step()
            return loss.detach().cpu().item()

        # Online learning
        if self.online and targ != self.inv_class:
            self.model.train()
            loss = do_update(hist_in, targ)

        # Update window
        self.hist[self.h_tl] = x
        self.h_tl = (self.h_tl + 1) % self.win

    def predict(self, k=1):
        if self.filled and self.online:
            self.model.eval()
            
            # Create input history
            hist_in = self.hist_idx()
            hist_in = torch.tensor(np.roll(hist_in, -self.h_tl))
            prev_addr = self.hist[(self.h_tl - 1) % self.win]
            
            if self.win != None and not self.filled:
                return [(0, 0)]
            model_out = self.model(hist_in.to(self.device))
            
            # Reverse map indices to addresses
            def ind(x, c):
                delta = self.inv_class - x
                if delta <= 2*self.n_d and delta != 0:
                    delta = -delta + self.n_d
                    if delta >= 0:
                        delta += 1
                    
                    self.delta_preds += 1
                    return prev_addr + delta, c
                elif x in self.oh.rmap:
                    p = self.oh.rmap[x]
                    self.addr_preds += 1
                    return p, c
                else:
                    return None, 0
                
            # Get prediction(s)
            if k == 1:
                item = model_out.squeeze().topk(1)
                preds = [ind(item[1].item(), item[0].item())]
                return preds
            else:
                out = model_out.squeeze().topk(k)
                preds = [ind(b.item(), a.item()) for a, b in zip(out[0], out[1])]
                return preds
        else:
            return []
        
    def print_parameters(self, base=""):
        print(f"{base}Address predictions:\t{self.addr_preds}")
        print(f"{base}Delta predictions:\t{self.delta_preds}")
        return