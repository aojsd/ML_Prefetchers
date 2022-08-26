import argparse
from parse import parse

parser = argparse.ArgumentParser()
parser.add_argument("infile", help="input file", type=str, default="")
parser.add_argument("tot", help="og missses", type=int)
args=parser.parse_args()

f = open(args.infile, 'r')
lines = f.readlines()

misses = parse("Total misses:\t{:d}\n", lines[1])[0]
total = parse("Total predictions:\t{:d}\n", lines[3])[0]
useful = parse("Useful prefetches:\t{:d}\n", lines[4])[0]
repeat = parse("Repeat prefetches:\t{:d}\n", lines[5])[0]
early = parse("Early prefetches:\t{:d}\n", lines[6])[0]
timeliness = parse("Complete Timeliness:\t{:f}\n", lines[8])[0]

acc = useful/(total-repeat)
cov = 1 - misses/args.tot

print("Accuracy:\t" + str(100*acc))
print("Coverage:\t" + str(100*cov))
print("Repeat %:\t" + str(100*repeat/total))
print("Early %:\t" + str(100*early/(total-repeat)))
print("Avg Time:\t" + str(timeliness))