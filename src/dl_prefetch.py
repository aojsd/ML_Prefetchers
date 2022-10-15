import random
import torch
import torch.nn.functional as F
from group_prefetcher import *

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
    def __init__(self, model, device, oh_params=None, online_params=None, lr=1e-3):
        self.model = model.to(device)
        self.device = device
        self.state = None
        self.oh = oh_params

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
        # For onehot model, convert addresses to indices
        if self.oh != None:
            if x in self.oh.imap:
                x = self.oh.imap[x]
            elif self.c_limit and self.c_alloced < self.inv_class:
                self.oh.imap[x] = self.c_alloced
                self.oh.rmap[self.c_alloced] = x

                x = self.c_alloced
                self.c_alloced += 1
            else:
                x = self.inv_class

        # Function for weight update
        #   Both args should be tensors
        def do_update(input, output, st):
            out, _ = self.model(input.to(self.device), tuple(s.to(self.device) for s in st))
            if len(out.shape) > 2:
                out = out.squeeze()
            loss = F.cross_entropy(out, output.to(self.device))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss.detach().cpu().item()

        # Online learning
        if self.online and not self.x == None and not x == self.inv_class:
            if not self.init_replay:
                self.model.train()
                loss = do_update(self.x, torch.tensor([x]), self.state)

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
                    # hs, ms, es = self.replay_mem.sample(int(tot/2), int(tot/3), int(tot/6))
                    hs, ms, es = self.replay_mem.sample(0, 0, tot)

                    # Do replay learning                    
                    # iterate_replay(hs, self.replay_mem.hard)
                    # iterate_replay(ms, self.replay_mem.med)
                    iterate_replay(es, self.replay_mem.easy)

                    # for i in hs:
                    #     x_in, targ, state = self.replay_mem[i]
                    #     out, new_st = self.model(x_in.to(self.device), tuple(s.to(self.device) for s in state))
                    #     loss = F.cross_entropy(out, targ.to(self.device))
                    #     self.optimizer.zero_grad()
                    #     loss.backward()
                    #     self.optimizer.step()

                        # if i+1 < len(self.replay_mem):
                        #     x_r, t_r, _ = self.replay_mem[i+1]
                        #     self.replay_mem[i+1] = (x_r, t_r, tuple(s.detach().cpu() for s in new_st))

                # # Calculate loss mean and std (for selective replay)
                # if self.loss_vec == None:
                #     self.loss_vec = torch.tensor([loss])
                # else:
                #     self.loss_vec = torch.cat([self.loss_vec, torch.tensor([loss])])
                # mean = self.loss_vec.mean().item()
                # std = self.loss_vec.std().item()

                # Add to replay memory with some probability
                sample = (self.x, torch.tensor([x]), tuple(s.cpu() for s in self.state))
                # diff = loss - mean
                # if diff >= std:
                #     self.replay_mem.hard.append(sample)
                # elif diff >= .5*std:
                #     self.replay_mem.med.append(sample)
                # # elif abs(diff) <= .5*std:
                # else:
                if self.online_params.add:
                    if self.online_params.clear and not self.init_replay:
                        self.other_mem.easy.append(sample)
                    self.replay_mem.easy.append(sample)
                # self.replay_mem.append((self.x, torch.tensor([x]), tuple(s.cpu() for s in self.state)))

        self.x = torch.tensor([x])

    def predict(self, k=1):
        if self.online:
            self.model.eval()
        if self.oh != None:
            model_out, state = self.model(self.x.to(self.device), self.state)
            self.state = tuple([s.detach() for s in list(state)])
            
            # Reverse map indices to addresses
            def ind(x):
                if x in self.oh.rmap:
                    return self.oh.rmap[x], 1
                else:
                    return None, 0
            if k == 1:
                return [ind(model_out.argmax().item())]
            else:
                out = model_out.topk(k, dim=-1)
                preds = out[1].squeeze().tolist()
                return [ind(p) for p in preds]
        else:
            model_out, self.state = self.model.predict(self.x, self.state)
            out = unsplit(model_out, self.model.splits, self.model.len_split)
            return [out.item(), 1]
        
    def print_parameters(self):
        return