import gzip
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collect_trace import process_access_trace

parser = argparse.ArgumentParser()
parser.add_argument("infile", help="input file", type=str, default="")
parser.add_argument("outfile", help="output file", type=str, default="")
args=parser.parse_args()

def sim_trace(file, sz):
    input = gzip.open(file)
    ret = process_access_trace(input, -1, sz)
    input.close()
    return ret

cache_sizes = np.arange(2, 0, -0.125)
misses = [sim_trace(args.infile, s) for s in cache_sizes]

plt.plot(cache_sizes, misses)
plt.gca().invert_xaxis()
plt.title("Page Misses vs. Memory Size")
plt.ylabel("Misses")
plt.xlabel("Memory Size (GB)")
plt.savefig("figs/" + args.outfile)