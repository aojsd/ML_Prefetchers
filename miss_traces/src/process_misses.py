import argparse
import pandas as pd
import torch
from load import load_data_dir

parser = argparse.ArgumentParser()
parser.add_argument("infile", help="input file", type=str, default="")
parser.add_argument("-o", help="output file", type=str, default="")
parser.add_argument("--index", help="get trace of indices", action="store_true", default=False)
parser.add_argument("--dir", help="directory for multiple files", type=str, default="")
parser.add_argument("--pref", help="prefix for multiple files", type=str, default="")
parser.add_argument("--classes", help="number of classes", type=int, default=2500)
parser.add_argument("-r", help="check repeating addresses", action="store_true", default=False)
parser.add_argument("-i", help="check repeating instruction pointers", action="store_true", default=False)
parser.add_argument("-c", help="check for repeating chains", type=int, default=1)
parser.add_argument("-t", help="minimum # repeats threshold for repeating stats", type=int, default=1)
parser.add_argument("-w", help="flexible window", action="store_true", default=False)

parser.add_argument("--ip", help="use instruction pointers", action='store_true', default=False)
parser.add_argument("-d", help="convert to deltas between consecutive addresses", action="store_true", default=False)
parser.add_argument("--dataset", help="convert to dataset with outputs being next address", action="store_true", default=False)
args=parser.parse_args()

if args.d or args.dataset:
    trace = pd.read_csv(args.infile, header=None)
    if args.d:
        shifted = trace.iloc[1:].reset_index(drop=True)
        trace = trace.iloc[:-1]
        trace = shifted - trace
    if args.dataset:
        shifted = trace.iloc[1:]
        trace = trace.iloc[:-1]
        trace[1] = shifted.reset_index(drop=True)
    trace.to_csv(args.o, header=None, index=None)

if args.index:
    _, _, index_map, rev_map, _ = load_data_dir(args.dir, args.classes, args.pref)
    trace = pd.read_csv(args.infile, header=None)
    def transform(x):
        if x in index_map:
            return index_map[x]
        else:
            return args.classes
    
    trace = trace.applymap(transform)
    trace.to_csv(args.o, header=None, index=None)

if args.r:
    f = open(args.infile, "r")
    lines = f.readlines()
    f.close()
    
    # Setup initial window first
    window = []
    seen = {}
    rpt = {}
    tot = 0
    for _ in range(args.c):
        l = lines.pop(0)
        if not args.ip:
            p = int(l, 0)
        else:
            p = int(l.split()[2], 0)

        window.append(p)
        if p not in seen:
            seen[p] = [0]
    
    # Without args.w: match exact chains of length c
    if not args.w:
        chains = set()
        chains.add(tuple(window))
    
    # Iterate
    for l in lines:
        if not args.ip:
            p = int(l, 0)
        else:
            p = int(l.split()[2], 0)
        window.append(p)
        window.pop(0)
        
        # No flexible window -- exact chains
        if not args.w:
            ch = tuple(window)
            if ch in chains:
                tot += 1
                if ch in rpt:
                    rpt[ch] += 1
                else:
                    rpt[ch] = 1
            else: chains.add(ch)
            
        # Flexible window
        else:
            if p in seen:
                seen[p].append(0)
                # tot += 1
                # if p in rpt:
                #     for a in window:
                #         if a in rpt[p]:
                #             rpt[p][a] += 1
                #         else:
                #             rpt[p][a] = 1
                # else:
                #     rpt[p] = {}
                #     for a in window:
                #         rpt[p][a] = 1
            else: seen[p] = [0]
            
            for a in window:
                if a not in rpt:
                    rpt[a] = {}

                # p has been seen before soon after a
                if p in rpt[a]:
                    rpt[a][p] += 1
                    tot += 1

                # p is a new page with respect to a
                else:
                    rpt[a][p] = 1
                    seen[a][-1] += 1


    rpts = []
    tgts = []
    maxs = []
    means = []
    med = []
    # Get average number of times each chain repeats
    if not args.w:
        ch_pages = set()
        for ch in rpt:
            if rpt[ch] >= args.t:
                rpts.append(rpt[ch])
                for p in ch:
                    ch_pages.add(p)
        x = torch.tensor(rpts, dtype=torch.float)
    else:
        p_tot = 0
        for a in rpt:
            a_len = len(seen[a])
            if a_len > args.t:
                p_tot += 1
                # Rpts stores # of pages that are correlated with the future of a divided by the repetitions of a
                tgts.append(len(rpt[a]))
                rpts.append(sum(seen[a])/a_len)
                
                # x stores the number of times a is correlated with any other page
                x = []
                for p in rpt[a]:
                    x.append(rpt[a][p])
                x = torch.tensor(x, dtype=torch.float)
                maxs.append(torch.max(x).item()/len(seen[a]))
                means.append(torch.mean(x).item()/len(seen[a]))
                med.append(torch.median(x).item()/len(seen[a]))
        
    print("\nMisses: ", len(lines) + args.c)
    
    if not args.w:
        print("Total Repetitions of Pages/Chains: ", tot)
        print()
        print("Chain Length: ", args.c)
        print("Number of Chains: ", len(chains))
        print(f"Repeating Chains (Threshold = {args.t}): {len(rpts)}")
        print("Pages in Repeating Chains: ", len(ch_pages))
        print("Mean Repeats: ", torch.mean(x).item())
        print("Max Repeats: ", torch.max(x).item())
        print("Median Repeats: ", torch.median(x).item())
        print("Std Repeats: ", torch.std(x).item())
    else:
        rpts = torch.tensor(rpts, dtype=torch.float)
        tgts = torch.tensor(tgts, dtype=torch.float)
        maxs = torch.tensor(maxs, dtype=torch.float)
        means = torch.tensor(means, dtype=torch.float)
        med = torch.tensor(med, dtype=torch.float)
        print("Number of Correlation Repetitions: ", tot)
        print()
        print("Window Size: ", args.c)
        print("Repeat Threshold: ", args.t)
        print("Pages over Threshold: ", p_tot)
        print("Mean # Targets: ", tgts.mean().item())
        print("Min # Targets: ", tgts.min().item())
        print("Mean New Targets/Rep: ", rpts.mean().item())
        # print("Median Avg Targets/Rep: ", rpts.median().item())
        # print("Std Targets/Rep: ", rpts.std().item())
        print()
        print("Mean Max Correlation/Rep: ", maxs.mean().item())
        print("Max Max Correlation/Rep: ", maxs.max().item())
        print("Median Max Correlation/Rep: ", maxs.median().item())
        # print("Std Max Correlation: ", maxs.std().item())
        print()
        print("Mean Mean Correlation/Rep: ", means.mean().item())
        print("Mean Median Correlation/Rep: ", med.mean().item())
        print("Median Median Correlation/Rep: ", med.median().item())
        print("25% Median Correlation/Rep: ", med.quantile(0.25).item())
        # print("Max Mean Correlation/Rep: ", means.max().item())
        # print("Median Mean Correlation/Rep: ", means.median().item())
        # print("Std Mean Correlation: ", means.std().item())
        