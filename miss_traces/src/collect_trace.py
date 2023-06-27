import gzip
import time
import argparse

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
def process_access_trace(input, n, buffer_size, use_ip=False, trace_misses=False, outfile="misses.txt"):
    # Output miss trace
    if trace_misses:
        out = open(outfile, 'w+')
    
    # Initialize prefetch window and stats
    n_lines = 0
    num_misses = 0

    # Memory buffer
    mem_buf = MemoryBuffer(int(buffer_size * (1 << 18)))

    # Function for extracting VPN from file
    def get_VPN(line):
        data = line.strip()
        miss_addr = int(data, 0)
        vpn = miss_addr >> 12
        return vpn

    # Start simulating memory access trace
    while n_lines < n or n == -1:
        line = input.readline()
        if not line:
            break
        n_lines += 1
        
        if use_ip:
            parsed = line.split()
            ip  = int(parsed[0], 0)
            acc_t = parsed[1]
            vpn = int(parsed[2], 0) >> 12
        else:
            vpn = get_VPN(line)

        # Check for page miss
        if not mem_buf.vpn_in(vpn):
            mem_buf.insert_page(vpn)
            num_misses += 1

            if trace_misses:
                if use_ip:
                    out.write(str(ip) + ' ' + acc_t + ' ' + str(vpn) + '\n')
                else:
                    out.write(str(vpn) + '\n')

    return num_misses

def main(args):
    # Run simulation
    input = gzip.open(args.infile, 'r')
    if args.o != None:
        misses = process_access_trace(input, args.n, args.b, args.ip, True, args.o)
    misses = process_access_trace(input, args.n, args.b, args.ip)
    print("Total misses: " + str(misses))
    print("Cache Size: " + str(args.b) + "GB")
    input.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="input file", type=str, default="")
    parser.add_argument("-o", help="output file", type=str, default=None)
    parser.add_argument("-b", help="buffer size in GB", type=float, default=1)
    parser.add_argument("-n", help="accesses to simulate", type=int, default=-1)
    parser.add_argument("--ip", help="use instruction pointers", action='store_true', default=False)

    # Timer
    start = time.time()

    args=parser.parse_args()
    main(args)

    print("\nSimulation took %s seconds" % (time.time() - start))