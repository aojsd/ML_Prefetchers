import os
import pandas as pd
import torch

# Merge data from all csv files in a directory
def load_data_dir(dir, max_classes, prefix="", batch_size=128, nrows=-1):
    # Things to return
    data_iters = []
    iter_sizes = []
    index_map = {}
    rev_map = {}

    # Read in all file data
    d_frames = []
    freqs = []
    csvs = []
    temp_index = {}
    temp_rev = {}
    count = 0
    for f in os.listdir(dir):
        if f.startswith(prefix) and f.endswith(".csv"):
            fname = dir + "/" + f
            if nrows != -1:
                data = pd.read_csv(fname, nrows=nrows, header=None)
            else:
                data = pd.read_csv(fname, header=None)
            csvs.append(data)
            iter_sizes.append(len(data.index))

    min_sz = min(iter_sizes)
    for data, sz in zip(csvs, iter_sizes):
        # Get unique values and their frequencies, weighted by sequence length
        if max_classes != None:
            f_weight = min_sz/sz
            for d in data[0]:
                if d not in temp_index:
                    temp_index[d] = count
                    temp_rev[count] = d
                    freqs.append(f_weight)
                    count += 1
                else:
                    freqs[temp_index[d]] += f_weight

        d_frames.append(data)

    # Produce index and reverse maps for only the most frequent values
    if max_classes != None:
        if len(freqs) > max_classes:
            count = 0
            top_indices = torch.topk(torch.Tensor(freqs), max_classes)[1].tolist()
            for i in top_indices:
                v = temp_rev[i]
                index_map[v] = count
                rev_map[count] = v
                count += 1
            
            del temp_index
            del temp_rev
        else:
            max_classes = count
            index_map = temp_index
            rev_map = temp_rev

        # Function to replace data with indices instead of raw values, infrequent values set to invalid index
        def index_replace(x):
            if x in index_map:
                return index_map[x]
            else:
                return max_classes

    # Setup data loaders from dataframes
    for data in d_frames:
        if max_classes != None:
            data = data.applymap(index_replace)
        addr_in = torch.tensor(data[0].to_numpy())
        targets = torch.tensor(data[1].to_numpy())

        d_set  = torch.utils.data.TensorDataset(addr_in, targets)
        d_iter = torch.utils.data.DataLoader(d_set, batch_size, shuffle=False)
        data_iters.append(d_iter)

        del addr_in
        del targets
        del data
    del d_frames

    return data_iters, iter_sizes, index_map, rev_map, max_classes