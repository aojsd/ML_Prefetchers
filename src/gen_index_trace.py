import argparse
import os
import pandas as pd
from multiset_onehot import load_data_dir

def main(args):
    if args.indir == None:
        data = pd.read_csv(args.infile, header=None)
        index_map = {}
        count = 0
        for d in data[0]:
            if d not in index_map:
                index_map[d] = count
                count += 1
        data = data.applymap(lambda x: index_map[x])
        data.to_csv(args.outfile, header=False, index=False)
    else:
        _, _, index_map, _, _ = load_data_dir(args.indir, args.max_classes)
        def index_replace(x):
            if x in index_map:
                return index_map[x]
            else:
                return args.max_classes
        
        data = pd.read_csv(args.infile, header=None)
        data = data.applymap(index_replace)
        data.to_csv(args.outfile, header=False, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str)
    parser.add_argument("outfile", type=str)
    parser.add_argument("--indir", help="Directory for combined traces (frequencies)", type=str, default=None)
    parser.add_argument("--max_classes", type=int, default=20000)

    args = parser.parse_args()
    main(args)