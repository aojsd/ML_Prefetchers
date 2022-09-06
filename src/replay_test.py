# Imports
import gzip
import time
import argparse
import torch
import pandas as pd
from one_hot_pages_model import OneHotNet
from multiset_onehot import load_data_dir
from dl_prefetch import DLPrefetcher
from prefetch_sim import *

# Initialize replay memory from target file(s)
# def init_replay(dir, prefix, pref : DLPrefetcher):
#     for f in os.listdir(dir):
#         if f.startswith(prefix):
#             fname = dir + "/" + f
#             break
    
#     dat = pd.read_csv(fname, header=None)[0].values.tolist()
#     deltas = [int(l) for l in dat]

#     replay_mem = []
#     state = None
#     last_in = None
#     pref.model.eval()
#     for d in deltas:
#         if d in pref.oh.imap:
#             d = pref.oh.imap[d]
#         else:
#             d = pref.inv_class
#         d = torch.tensor([d])

#         _, state = pref.model(d.to(pref.device), state)

#         if last_in != None:
#             replay_mem.append((last_in, d, tuple(s.detach().cpu() for s in state)))
#         last_in = d

#     return replay_mem


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
        _, _, index_map, rev_map, _ = load_data_dir(args.miss_file, args.max_classes, prefix=args.prefix)
        classes = args.max_classes
        oh_params = OnehotParams(index_map, rev_map)
    else:
        assert args.max_classes != 0
        index_map = {}
        rev_map = {}
        classes = args.max_classes
        limit = classes
        oh_params = OnehotParams(index_map, rev_map, limit)

    # LSTM Parameters
    e_dim = 32
    h_dim = 32
    layers = 1
    dropout = 0.1

    # Training params
    lr = args.lr
    online_params = None
    if args.online:
        if args.replay:
            online_params = OnlineParams(True)
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
    if args.replay:
        input = gzip.open(lines[-1].strip(), 'r')
        misses, fetches, useful, repeats, early, use_time, comp_time = pref_raw_trace(
            input, prefetcher, 0, args.n, args.b, init=False,
            tagging=tagging, stream=False, raw=False, k=args.k
        )
        input.close()

        # Reset loss list to have per-trace mean and std
        prefetcher.loss_vec = None
        prefetcher.count = 0

    for trace in lines:
        input = gzip.open(trace.strip(), 'r')
        misses, fetches, useful, repeats, early, use_time, comp_time = pref_raw_trace(
            input, prefetcher, 0, args.n, args.b, init=False,
            tagging=tagging, stream=False, raw=False, k=args.k
        )
        prefetcher.loss_vec = None
        prefetcher.count = 0

        print("\n" + trace.strip())
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

    print()
    print("Embedding Size:\t" + str(e_dim))
    print("Hidden Size:\t" + str(h_dim))
    print("Classes:\t\t" + str(classes))
    print("Learning Rate:\t" + str(lr))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("infiles", help="input file", type=str, default="")
    parser.add_argument("-n", help="accesses to simulate", type=int, default=-1)
    parser.add_argument("-b", help="buffer size in GB", type=float, default=1)
    parser.add_argument("-t", help="tag prefetched pages for more predictions", action='store_true', default=False)
    parser.add_argument("-k", help="number of predictions to make", type=int, default=1)
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
    