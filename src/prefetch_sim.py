# Associative Recall-Based Page Prefetcher
#   Uses an autoassociative network as the core pattern learning and
#   recall (recognition) structure
#   Takes a stream of page/TLB misses and predicts the next miss VPN
#   after a short warmup period

# Imports
import gzip
from textwrap import indent
import time
import argparse
import random
import torch
import torch.nn.functional as F
from group_prefetcher import *
from one_hot_pages_model import OneHotNet
from one_hot_pages_model import load_data
from multiset_onehot import load_data_dir

# Parameters for onehot models
class OnehotParams():
    def __init__(self, imap, rmap, class_limit=None):
        self.imap = imap
        self.rmap = rmap
        self.class_limit = class_limit

# Parameters for online training
class OnlineParams():
    def __init__(self, replay=False, replay_len=5, replay_freq=5):
        self.replay = replay
        self.rep_len = replay_len
        self.rep_fr = replay_freq

# Deep Learning Prefetcher
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
        if online_params != None:
            self.online = True
            self.online_params = online_params
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

            self.replay_mem = None
            if online_params.replay:
                self.replay_mem = []
                self.rep_count = 0
                self.rep_empty = True
        self.x = None

    def learn(self, x):
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

        if self.online and not self.x == None and not x == self.inv_class:
            self.model.train()
            out, _ = self.model(self.x, self.state)
            loss = F.cross_entropy(out, torch.tensor([x]).to(self.device))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Interleaved learning using replay memory
            if self.replay_mem != None:
                # Replay learning
                if not self.rep_empty or len(self.replay_mem) > self.online_params.rep_len:
                    self.rep_empty = False
                    
                    if self.rep_count % self.online_params.rep_fr == 0:
                            
                        i_list = random.sample(list(range(len(self.replay_mem))), self.online_params.rep_len)
                        # ind_end = int(self.online_params.rep_len * len(self.replay_mem))
                        # r_list = self.replay_mem[:ind_end]
                        for i in i_list:
                            x_in, targ, state = self.replay_mem[i]
                            out, new_st = self.model(x_in, tuple(s.to(self.device) for s in state))
                            loss = F.cross_entropy(out, torch.tensor([targ]).to(self.device))
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                            if i+1 < len(self.replay_mem):
                                nx, nt, _ = self.replay_mem[i+1]
                                self.replay_mem[i+1] = (nx, nt, tuple(s.detach().cpu() for s in new_st))
                    self.rep_count += 1

                # Add to replay memory with some probability   
                # if random.uniform(0,1) < self.online_params.s_pr:
                self.replay_mem.append((self.x, x, tuple(s.cpu() for s in self.state)))

        self.x = torch.tensor([x]).to(self.device)

    def predict(self):
        if self.online:
            self.model.eval()
        if self.oh != None:
            model_out, state = self.model(self.x, self.state)
            self.state = tuple([s.detach() for s in list(state)])
            out = model_out.argmax(dim=-1)
            
            pred = out.item()
            if pred in self.oh.rmap:
                return self.oh.rmap[out.item()], 1
            else:
                return 0, 0
        else:
            model_out, self.state = self.model.predict(self.x, self.state)
            out = unsplit(model_out, self.model.splits, self.model.len_split)
            return out.item(), 1
        
    def print_parameters(self):
        return

# Fifo memory buffer for pages
class MemoryBuffer():
    def __init__(self, buffer_size):
        self.max_pages = int(buffer_size)
        self.buffer = set()
        self.fifo = [None for _ in range(self.max_pages)]
        self.head = 0
        self.full = False
    
    def insert_page(self, vpn):
        # Evict a page if the buffer is full
        evicted = None
        if self.full:
            evicted = self.fifo[self.head]
            self.buffer.remove(evicted)
            self.buffer.add(vpn)
            self.fifo[self.head] = vpn
            self.head = (self.head + 1) % self.max_pages
        else:
            self.buffer.add(vpn)
            self.fifo[self.head] = vpn
            self.head = (self.head + 1) % self.max_pages

            # Full check
            if self.head == 0:
                self.full = True
        return evicted

    def vpn_in(self, vpn):
        return (vpn in self.buffer)

# Run prefetcher on a raw memory access trace
def pref_raw_trace(input, prefetcher, skip, n, buffer_size, tagging=True, init=True,
                   stream=False, raw=False, repeat_trace=None):
    # Function for extracting VPN from file
    def get_VPN(line):
        data = line.strip()
        miss_addr = int(data, 0)
        vpn = miss_addr >> 12
        return vpn

    # Memory buffer
    mem_buf = MemoryBuffer(int(buffer_size * (1 << 18)))

    # Skip lines
    for _ in range(skip):
        input.readline()

    # Initialize prefetch window and stats
    n_lines = 0
    num_misses = 0
    num_fetches = 0
    num_useful = 0
    num_repeats = 0
    num_unused = 0
    init_buf = []
    last_miss_vpn = None
    base_vpn = None

    # Initialize buffer for window-based models
    if init:
        inserts = 0
        while inserts < prefetcher.window_size:
            line = input.readline()
            if not line:
                break
            n_lines += 1

            # Get VPN from format and hex
            vpn = get_VPN(line)

            # Check for miss in local memory, cannot be full in this loop
            if not mem_buf.vpn_in(vpn):
                mem_buf.insert_page(vpn)

                # Add to prefetch buffer using one-hop distances
                if not stream and not raw:
                    if last_miss_vpn != None:
                        delta = vpn - last_miss_vpn
                        last_miss_vpn = vpn
                    else:
                        last_miss_vpn = vpn
                        delta = vpn
                    init_buf.append(delta)
                    inserts += 1
                # Use raw addresses
                elif raw:
                    init_buf.append(vpn)
                    inserts += 1
                # Use relative addressing windows
                elif stream:
                    if base_vpn == None:
                        base_vpn = vpn
                    else:
                        delta = vpn - base_vpn
                        init_buf.append(delta)
                        inserts += 1
                num_misses += 1

        # Enough misses to initialize prefetch buffer
        prefetcher.initialize_buffer(init_buf)
    else:
        last_miss_vpn = 0

    # Start prefetching on memory access trace
    prefetched_vpns = set()
    while n_lines < n or n == -1:
        line = input.readline()
        if not line:
            break
        n_lines += 1
        vpn = get_VPN(line)

        # Check for page miss
        if not mem_buf.vpn_in(vpn):
            evicted = mem_buf.insert_page(vpn)
            if evicted in prefetched_vpns:
                num_unused += 1
                prefetched_vpns.remove(evicted)

            # Prefetch logic start
            num_misses += 1

            # Learn off of miss delta or address
            if not stream and not raw:
                delta = vpn - last_miss_vpn
                last_miss_vpn = vpn
                prefetcher.learn(delta)
            elif raw:
                prefetcher.learn(vpn)
            elif stream:
                base_vpn += prefetcher.get_buf_entry(0)
                prefetcher.base_update()
                prefetcher.learn(vpn - base_vpn)

            # Make prediction
            pred, rec = prefetcher.predict()
            if pred != None:
                num_fetches += 1
                
                # Add result to vpn if using deltas, to base_vpn if relative addressing
                if not stream and not raw:
                    pred += vpn
                elif stream:
                    pred += base_vpn

                # Track prediction and add to memory
                if not mem_buf.vpn_in(pred):
                    prefetched_vpns.add(pred)
                    mem_buf.insert_page(pred)
                else:
                    # Track repeated prefetches (prediction already in memory)
                    num_repeats += 1

                    # Gather data
                    if repeat_trace != None:
                        w_str = "pred " + str(num_misses) + ":" + "\t"
                        if not stream and not raw:
                            w_str += "d: " + str(pred - vpn)
                        elif stream:
                            w_str += str(pred - base_vpn)

        # If page hit
        else:
            # Check if a prefetch is useful (used at least once)
            if vpn in prefetched_vpns:
                # increment useful only on new use
                num_useful += 1
                prefetched_vpns.remove(vpn)

                # Learn and predict on a useful prefetch (should test with or without)
                if tagging:
                    if not stream and not raw:
                        delta = vpn - last_miss_vpn
                        last_miss_vpn = vpn
                        prefetcher.learn(delta)
                    elif raw:
                        prefetcher.learn(vpn)
                    else:
                        base_vpn += prefetcher.get_buf_entry(0)
                        prefetcher.base_update()
                        prefetcher.learn(vpn - base_vpn)

                    pred, rec = prefetcher.predict()
                    if pred != None:
                        num_fetches += 1
                        if not stream and not raw:
                            pred += vpn
                        elif stream:
                            pred += base_vpn

                        if not mem_buf.vpn_in(pred):
                            prefetched_vpns.add(pred)
                            mem_buf.insert_page(pred)
                        else:
                            num_repeats += 1

                            # Gather data
                            if repeat_trace != None:
                                w_str = "pred " + str(num_misses) + ":" + "\t"
                                if not stream and not raw:
                                    w_str += "d: " + str(pred - vpn)
                                elif stream:
                                    w_str += str(pred - base_vpn)

    num_unused += len(prefetched_vpns)
    return num_misses, num_fetches, num_useful, num_repeats, num_unused

def main(args):
    if not args.nocuda:
        device = torch.device('cuda:0')
    else:
        device = 'cpu'
    
    # Prefetch parameters
    tagging = args.t != 0

    if args.oh or args.comb:
        # Get parameters and index 
        if args.comb:
            _, _, index_map, rev_map, _ = load_data_dir(args.miss_file, args.max_classes)
            classes = args.max_classes
            oh_params = OnehotParams(index_map, rev_map)
        if args.miss_file != None:
            _, _, index_map, rev_map, classes = load_data(args.miss_file)
            oh_params = OnehotParams(index_map, rev_map)
        else:
            assert args.max_classes != 0
            index_map = {}
            rev_map = {}
            classes = args.max_classes
            limit = classes
            oh_params = OnehotParams(index_map, rev_map, limit)

        # Online training parameters
        online_params = None
        if args.online:
            if args.replay:
                online_params = OnlineParams(True)
            else:
                online_params = OnlineParams(False)

        # LSTM Parameters
        e_dim = 256
        h_dim = 256
        layers = 1
        dropout = 0.1

        # Create net
        net = OneHotNet(classes, e_dim, h_dim, layers, dropout=dropout)
        prefetcher = DLPrefetcher(net, device, oh_params=oh_params, online_params=online_params)
    else:
        # Model parameters
        num_bits = 36
        splits = 4
        e_dim = 64
        h_dim = 128
        layers = 1
        dropout = 0.1

        # Create net
        net = SplitBinaryNet(num_bits, e_dim, h_dim, layers, splits=splits, dropout=dropout)
        prefetcher = DLPrefetcher(net, device)

    # Check for model file
    if args.model_file != None:
        if os.path.exists(args.model_file):
            print("Loading model from file: {}".format(args.model_file))
            net.load_state_dict(torch.load(args.model_file))
        else:
            if not args.online:
                print("Error: no model file given")
                exit()

    # Run simulation
    input = gzip.open(args.infile, 'r')
    misses, fetches, useful, repeats, unused = pref_raw_trace(input, prefetcher, args.s, args.n, args.b,
                                                    init=False, tagging=tagging, stream=False, raw=False)
    print("Total misses: " + str(misses))
    print("Cache Size: " + str(args.b) + "GB")
    print("Total predictions: " + str(fetches))
    print("Useful prefetches: " + str(useful))
    print("Repeat prefetches: " + str(repeats))
    print("Unused prefetches: " + str(unused))
    print("Useful percentage: %.3f" % (useful*100/fetches))
    print("Repeat percentage: %.3f" % (repeats*100/fetches))
    print("\tTagging: " + str(tagging))

    if args.oh:
        print()
        print("Embedding Size:\t" + str(e_dim))
        print("Hidden Size:\t" + str(h_dim))
        print("Classes:\t\t" + str(classes))

    input.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="input file", type=str, default="")
    parser.add_argument("-s", help="memory accesses to skip", type=int, default=0)
    parser.add_argument("-n", help="accesses to simulate", type=int, default=-1)
    parser.add_argument("-b", help="buffer size in GB", type=float, default=1)
    parser.add_argument("-t", help="tag prefetched pages for more predictions", action='store_true', default=False)
    parser.add_argument("--oh", help="Use one-hot model", action="store_true", default=False)
    parser.add_argument("--comb", help="Use combined traces for one-hot model, should be used with --max_classes", action="store_true", default=False)
    parser.add_argument("--online", help="Online training", action="store_true", default=False)
    parser.add_argument("--replay", help="Replay parameters for online training", action='store_true', default=False)
    parser.add_argument("--nocuda", help="Don't use cuda", action="store_true", default=False)
    parser.add_argument("--max_classes", help="Max classes for addresses/deltas", type=int, default=20000)
    parser.add_argument("--model_file", help="File to load/save model parameters to continue training", default=None, type=str)
    parser.add_argument("--miss_file", help="File to get indices for delta classes (onehot model only)", default=None, type=str)

    # Timer
    start = time.time()

    args=parser.parse_args()
    main(args)

    print("\nSimulation took %s seconds" % (time.time() - start))
    