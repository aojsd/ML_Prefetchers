# Imports
import gzip
import time
import argparse
import torch
from dl_prefetch import *
from group_prefetcher import SplitBinaryNet
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
    def __init__(self, replay=False, replay_len=100, replay_freq=50):
        self.replay = replay
        self.rep_len = replay_len
        self.rep_fr = replay_freq
        self.add = True
        self.clear = False

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
                   stream=False, raw=False, repeat_trace=None, k=1):
    # Initialize prefetch window and stats
    n_lines = 0
    num_misses = 0
    num_fetches = 0
    num_useful = 0
    num_repeats = 0
    num_early = 0
    useful_timeliness = 0
    complete_timeliness = 0
    init_buf = []
    last_miss_vpn = None
    base_vpn = None

    # Memory buffer
    mem_buf = MemoryBuffer(int(buffer_size * (1 << 18)))

    # Prefetch lists
    prefetch_list = {}
    evicted_preds = {}

    # Function for extracting VPN from file
    def get_VPN(line):
        data = line.strip()
        miss_addr = int(data, 0)
        vpn = miss_addr >> 12
        return vpn

    # Prefetch logic function
    def do_prefetch(n_fetches, n_repeats, n_misses):
        # Make prediction
        for pred, rec in prefetcher.predict(k):
            if pred != None:
                n_fetches += 1
                
                # Add result to vpn if using deltas, to base_vpn if relative addressing
                if not stream and not raw:
                    pred += vpn
                elif stream:
                    pred += base_vpn

                # Track prediction and add to memory
                if not mem_buf.vpn_in(pred):
                    prefetch_list[pred] = n_misses

                    # Re-prefetched, remove from evicted prefetches list
                    evicted_preds.pop(pred, None)

                    # Eviction tracking
                    evicted = mem_buf.insert_page(pred)
                    if evicted in prefetch_list:
                        evicted_preds[evicted] = prefetch_list.pop(evicted)
                        
                else:
                    # Track repeated prefetches (prediction already in memory)
                    n_repeats += 1

                    # Gather data
                    # if repeat_trace != None:
                    #     w_str = "pred " + str(num_misses) + ":" + "\t"
                    #     if not stream and not raw:
                    #         w_str += "d: " + str(pred - vpn)
                    #     elif stream:
                    #         w_str += str(pred - base_vpn)
        return n_fetches, n_repeats

    # Prefetch learning function
    def do_prefetcher_learn(addr, last_miss_addr, base_addr):
        if not stream and not raw:
            delta = addr - last_miss_addr
            last_miss_addr = addr
            prefetcher.learn(delta)
        elif raw:
            prefetcher.learn(addr)
        elif stream:
            base_addr += prefetcher.get_buf_entry(0)
            prefetcher.base_update()
            prefetcher.learn(addr - base_addr)
        return last_miss_addr, base_addr

    # Skip lines
    for _ in range(skip):
        input.readline()

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
    while n_lines < n or n == -1:
        line = input.readline()
        if not line:
            break
        n_lines += 1
        vpn = get_VPN(line)

        # If page miss
        if not mem_buf.vpn_in(vpn):
            num_misses += 1

            # Check for early prefetch
            if vpn in evicted_preds:
                num_early += 1
                complete_timeliness += num_misses - evicted_preds[vpn]
                del evicted_preds[vpn]

            # Eviction tracking
            evicted = mem_buf.insert_page(vpn)
            if evicted in prefetch_list:
                evicted_preds[evicted] = prefetch_list.pop(evicted)
            
            # Learn and predict
            last_miss_vpn, base_vpn = do_prefetcher_learn(vpn, last_miss_vpn, base_vpn)
            num_fetches, num_repeats = do_prefetch(num_fetches, num_repeats, num_misses)

        # If page hit
        else:
            # Check if a prefetch is useful (used at least once)
            if vpn in prefetch_list:
                # increment useful only on new use
                num_useful += 1
                diff = num_misses - prefetch_list.pop(vpn)
                complete_timeliness += diff
                useful_timeliness += diff

                # Learn and predict on a useful prefetch (should test with and without)
                if tagging:
                    last_miss_vpn, base_vpn = do_prefetcher_learn(vpn, last_miss_vpn, base_vpn)
                    num_fetches, num_repeats = do_prefetch(num_fetches, num_repeats, num_misses)
            
    # Timeliness avg
    useful_timeliness /= num_useful
    complete_timeliness /= num_useful + num_early

    return num_misses, num_fetches, num_useful, num_repeats, num_early, useful_timeliness, complete_timeliness

def main(args):
    torch.manual_seed(0)
    if not args.nocuda:
        device = torch.device('cuda:0')
    else:
        device = 'cpu'
    
    # Prefetch parameters
    tagging = args.t != 0

    # Online training parameters
    online_params = None
    if args.online:
        if args.replay_learn:
            online_params = OnlineParams(True)
        else:
            online_params = OnlineParams(False)

    if args.oh or args.comb:
        # Get parameters and index 
        if args.comb:
            _, _, index_map, rev_map, _ = load_data_dir(args.miss_file, args.max_classes, prefix=args.prefix)
            classes = args.max_classes
            oh_params = OnehotParams(index_map, rev_map)
        elif args.miss_file != None:
            _, _, index_map, rev_map, classes = load_data(args.miss_file)
            oh_params = OnehotParams(index_map, rev_map)
        else:
            assert args.max_classes != 0
            index_map = {}
            rev_map = {}
            classes = args.max_classes
            limit = classes
            oh_params = OnehotParams(index_map, rev_map, limit)

        # LSTM Parameters
        e_dim = 256
        h_dim = 256
        layers = 1
        dropout = 0.1

        # Create net
        net = OneHotNet(classes, e_dim, h_dim, layers, dropout=dropout)
        prefetcher = DLPrefetcher(net, device, oh_params=oh_params, online_params=online_params)

        print("Onehot Network")
        print("\tEH dim = " + str(e_dim))
    else:
        # Model parameters
        num_bits = 36
        splits = 12
        e_dim = 256
        h_dim = 256
        layers = 1
        dropout = 0.1

        # Create net
        net = SplitBinaryNet(num_bits, e_dim, h_dim, layers, splits=splits, dropout=dropout)
        prefetcher = SplitBinaryPref(net, device, online_params=online_params)

        print("Split Binary Network")
        print(f"\tBit Window = {int(num_bits/splits)}/36")
        print(f"\tEH Dim = {e_dim}")

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
    misses, fetches, useful, repeats, early, use_time, comp_time = pref_raw_trace(
        input, prefetcher, args.s, args.n, args.b, init=False,
        tagging=tagging, stream=False, raw=False, k=args.k
    )

    print("Total misses:\t" + str(misses))
    print("Cache Size:\t" + str(args.b) + "GB")
    print("Total predictions:\t" + str(fetches))
    print("Useful prefetches:\t" + str(useful))
    print("Repeat prefetches:\t" + str(repeats))
    print("Early prefetches:\t" + str(early))
    print("Useful Timeliness:\t" + str(use_time))
    print("Complete Timeliness:\t" + str(comp_time))
    print("\tTagging:\t" + str(tagging))
    print("\tPredictions per miss:\t" + str(args.k))
    input.close()

    if args.repeat:
        input = gzip.open(args.infile, 'r')
        prefetcher.state = None
        misses, fetches, useful, repeats, early, use_time, comp_time = pref_raw_trace(
            input, prefetcher, args.s, args.n, args.b, init=False,
            tagging=tagging, stream=False, raw=False
        )

        print("\n---------- Replay Run ----------")
        print("Total misses:\t" + str(misses))
        print("Cache Size:\t" + str(args.b) + "GB")
        print("Total predictions:\t" + str(fetches))
        print("Useful prefetches:\t" + str(useful))
        print("Repeat prefetches:\t" + str(repeats))
        print("Early prefetches:\t" + str(early))
        print("Useful Timeliness:\t" + str(use_time))
        print("Complete Timeliness:\t" + str(comp_time))
        print("\tTagging:\t" + str(tagging))
        print("\tPredictions per miss:\t" + str(args.k))
        input.close()

    if args.oh:
        print()
        print("Embedding Size:\t" + str(e_dim))
        print("Hidden Size:\t" + str(h_dim))
        print("Classes:\t\t" + str(classes))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="input file", type=str, default="")
    parser.add_argument("-s", help="memory accesses to skip", type=int, default=0)
    parser.add_argument("-n", help="accesses to simulate", type=int, default=-1)
    parser.add_argument("-b", help="buffer size in GB", type=float, default=2)
    parser.add_argument("-t", help="tag prefetched pages for more predictions", action='store_true', default=False)
    parser.add_argument("-k", help="number of predictions to make", type=int, default=1)
    parser.add_argument("--oh", help="Use one-hot model", action="store_true", default=False)
    parser.add_argument("--comb", help="Use combined traces for one-hot model, should be used with --max_classes and --prefix", action="store_true", default=False)
    parser.add_argument("--prefix", help="Prefix for combined models", type=str, default="")
    parser.add_argument("--online", help="Online training", action="store_true", default=False)
    parser.add_argument("--replay_learn", help="Use replay memory for online training", action='store_true', default=False)
    parser.add_argument("--repeat", help="Repeat simulation once (with training)", action='store_true', default=False)
    parser.add_argument("--nocuda", help="Don't use cuda", action="store_true", default=False)
    parser.add_argument("--max_classes", help="Max classes for addresses/deltas", type=int, default=20000)
    parser.add_argument("--model_file", help="File to load/save model parameters to continue training", default=None, type=str)
    parser.add_argument("--miss_file", help="File to get indices for delta classes (onehot model only)", default=None, type=str)

    # Timer
    start = time.time()

    args=parser.parse_args()
    main(args)

    print("\nSimulation took %s seconds" % (time.time() - start))
    