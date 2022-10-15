# Imports
import gzip
import time
import argparse
import torch
import pandas as pd
from one_hot_pages_model import OneHotNet
from multiset_onehot import load_data_dir
from dl_prefetch import DLPrefetcher, ReplayMemory
from prefetch_sim import *

def main(args):
    torch.manual_seed(0)
    if not args.nocuda:
        device = torch.device('cuda:0')
    else:
        device = 'cpu'
    
    # Prefetch parameters
    tagging = args.t != 0

    # Retrive model
    if args.comb:
        _, dset_sizes, index_map, rev_map, _ = load_data_dir(args.miss_file, args.max_classes, prefix=args.prefix)
        classes = args.max_classes
        oh_params = OnehotParams(index_map, rev_map)
        dset_sizes = [d/min(dset_sizes) for d in dset_sizes]
    else:
        assert args.max_classes != 0
        index_map = {}
        rev_map = {}
        classes = args.max_classes
        limit = classes
        oh_params = OnehotParams(index_map, rev_map, limit)

    # LSTM Parameters
    e_dim = args.eh
    h_dim = args.eh
    layers = 1
    dropout = 0.1

    # Training params
    lr = args.lr
    online_params = None
    if args.online:
        if args.replay:
            online_params = OnlineParams(True)
            
            # clear replay buffer between application runs
            online_params.clear = args.c
        else:
            online_params = OnlineParams(False)

    # Create net
    net = OneHotNet(classes, e_dim, h_dim, layers, dropout=dropout)
    prefetcher = DLPrefetcher(net, device, oh_params=oh_params, online_params=online_params, lr=lr)

    # Check for model file
    if args.model_file != None:
        if os.path.exists(args.model_file):
            print("Loading model from file: {}".format(args.model_file))
            net.load_state_dict(torch.load(args.model_file))
        else:
            if not args.online:
                print("Error: no model file given")
                exit()

    # See
    f = open(args.infiles, 'r')
    lines = f.readlines()

    # Initialize replay mem (selective by loss)
    print("Initializing Replay")
    if args.replay:
        line = lines[-1].split()
        input = gzip.open(line[0], 'r')
        prefetcher.init_replay = True
        wset_size = float(line[1])
        if args.b == None:
            buf_size = wset_size
        else: buf_size = args.b
        
        misses, fetches, useful, repeats, early, use_time, comp_time = pref_raw_trace(
            input, prefetcher, skip=0, n=args.n, buffer_size=buf_size, init=False,
            tagging=tagging, stream=False, raw=False, k=args.k
        )
        input.close()
        prefetcher.init_replay = False

        # Reset loss list to have per-trace mean and std
        prefetcher.loss_vec = None
        prefetcher.count = 0

    # Fix replay mem
    if args.f:
        print("\nFixing replay memory contents")
        prefetcher.online_params.add = False

    # Start simulations
    print("Start sim")
    for i, trace in enumerate(lines):
        # Stop replay for last iteration for fixed buffer
        if args.f and i == len(lines) - 1:
            print("\nStopping replay")
            prefetcher.online_params.replay = False

        info = trace.split()
        if args.b == None:
            b = float(info[1])
        else: b = args.b

        input = gzip.open(info[0], 'r')
        misses, fetches, useful, repeats, early, use_time, comp_time = pref_raw_trace(
            input, prefetcher, 0, args.n, b, init=False,
            tagging=tagging, stream=False, raw=False, k=args.k
        )
        prefetcher.loss_vec = None
        prefetcher.count = 0

        if args.c:
            prefetcher.replay_mem = prefetcher.other_mem
            prefetcher.other_mem = ReplayMemory()


        print("\n" + trace.strip())
        print("Total misses:\t" + str(misses))
        print("Cache Size:\t" + str(b) + "GB")
        print("Total predictions:\t" + str(fetches))
        print("Useful prefetches:\t" + str(useful))
        print("Repeat prefetches:\t" + str(repeats))
        print("Early prefetches:\t" + str(early))
        print("Useful Timeliness:\t" + str(use_time))
        print("Complete Timeliness:\t" + str(comp_time))
        print("\tTagging:\t" + str(tagging))
        print("\tPredictions per miss:\t" + str(args.k))
        input.close()

    print()
    print("Embedding Size:\t" + str(e_dim))
    print("Hidden Size:\t" + str(h_dim))
    print("Classes:\t\t" + str(classes))
    print("Learning Rate:\t" + str(lr))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("infiles", help="input file", type=str, default="")
    parser.add_argument("-n", help="accesses to simulate", type=int, default=-1)
    parser.add_argument("-t", help="tag prefetched pages for more predictions", action='store_true', default=False)
    parser.add_argument("-k", help="number of predictions to make", type=int, default=1)
    parser.add_argument("-f", help="fix buffer", action='store_true', default=False)
    parser.add_argument("-c", help="clear replay buffer between applications", action='store_true', default=False)
    parser.add_argument("-b", help="buffer size in GB", type=float, default=None)
    parser.add_argument("--eh", help="Embedding and hidden dimensions", type=int, default=32)
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-4)
    parser.add_argument("--comb", help="Use combined traces for one-hot model, should be used with --max_classes and --prefix", action="store_true", default=False)
    parser.add_argument("--prefix", help="Prefix for combined models", type=str, default="")
    parser.add_argument("--online", help="Online training", action="store_true", default=False)
    parser.add_argument("--replay", help="Use replay memory for online training", action='store_true', default=False)
    parser.add_argument("--nocuda", help="Don't use cuda", action="store_true", default=False)
    parser.add_argument("--max_classes", help="Max classes for addresses/deltas", type=int, default=20000)
    parser.add_argument("--model_file", help="File to load/save model parameters to continue training", default=None, type=str)
    parser.add_argument("--miss_file", help="File to get indices for delta classes (onehot model only)", default=None, type=str)

    # Timer
    start = time.time()

    args=parser.parse_args()
    main(args)

    print("\nSimulation took %s seconds" % (time.time() - start))
    