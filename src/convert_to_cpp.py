import argparse
import torch
import os
from one_hot_pages_model import OneHotNet
from one_hot_pages_model import load_data

def main(args):
    # Get parameters and index maps
    if args.miss_file != None:
        _, _, _, _, classes = load_data(args.miss_file)
    else:
        assert args.max_classes != 0
        classes = args.max_classes

    # Params
    e_dim = h_dim = args.eh
    layers = 1
    dropout = 0.1

    net = OneHotNet(classes, e_dim, h_dim, layers, dropout=dropout)

    # Check for model file
    if args.modelfile != None:
        if os.path.exists(args.modelfile):
            print("Loading model from file: {}".format(args.modelfile))
            net.load_state_dict(torch.load(args.modelfile))
        else:
            print('Model file does not exist')
            exit()

    # JIT model for cpp usage
    model_cpp = torch.jit.script(net)
    model_cpp.save(args.outfile)
    print(model_cpp.code)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("modelfile", help="file containing Pytorch model", type=str)
    parser.add_argument("outfile", help="file to write Torchscript model", type=str)
    parser.add_argument("--miss_file", help="File to get exact number of classes (onehot model only)", default=None, type=str)
    parser.add_argument('--max_classes', help='classes in embedding layer', type=int, default=0)
    parser.add_argument("--eh", help="Embedding and hidden dimensions", type=int, default=32)


    args = parser.parse_args()
    main(args)