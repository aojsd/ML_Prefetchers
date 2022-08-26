import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("infile", help="input file", type=str, default="")
parser.add_argument("outfile", help="output file", type=str, default="")
parser.add_argument("-d", help="convert to deltas between consecutive addresses", action="store_true", default=False)
parser.add_argument("--dataset", help="convert to dataset with outputs being next address", action="store_true", default=False)
args=parser.parse_args()

trace = pd.read_csv(args.infile, header=None)

if args.d:
    shifted = trace.iloc[1:].reset_index(drop=True)
    trace = trace.iloc[:-1]
    trace = shifted - trace

if args.dataset:
    shifted = trace.iloc[1:]
    trace = trace.iloc[:-1]
    trace[1] = shifted.reset_index(drop=True)

trace.to_csv(args.outfile, header=None, index=None)