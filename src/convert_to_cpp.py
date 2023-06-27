import argparse
import time
import os
from one_hot_pages_model import OneHotNet, load_data, setup_data, OHNet_eval, OHNet_train
from multiset_onehot import multiset_train, multiset_eval, load_data_dir
import torch
import torch.quantization

def main(args):
    # Get parameters and index maps
    if args.miss_file != None:
        if not args.dir:
            addr_in, targs, _, _, classes = load_data(args.miss_file)
            data_iter = setup_data(addr_in, targs, batch_size=args.b)
        else:
            data_iters, _, _, _, classes = load_data_dir(args.miss_file, args.max_classes, args.prefix, args.b, -1)
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
            
    # Quantize model
    if args.q:
        matmul = torch.quantization.get_default_qconfig('fbgemm')
        emb = torch.quantization.float_qparams_weight_only_qconfig
        torch.quantization.quantize_dynamic(net, {torch.nn.Linear: matmul, torch.nn.Embedding: emb, torch.nn.LSTM: matmul}, dtype=torch.qint8, inplace=True)

    # JIT model for cpp usage
    if not args.compile:
        model_cpp = torch.jit.script(net)
        model_cpp.save(args.out)
        print(model_cpp.code)
        
    # Compile model for timing
    else:
        # Set single-threaded
        torch.set_num_threads(1)
        
        # Compile model code
        opt_net = torch.compile(net, mode="max-autotune")
        print("Model compiled")
        
        # Inference
        if args.eval:
            if not args.dir:
                opt_eval = torch.compile(OHNet_eval, mode="max-autotune")
                print("Eval compiled")
                s = time.time()
                opt_eval(opt_net, data_iter)
                e = time.time()
                print("Eval time: {}".format((e-s)/len(data_iter)))
            else:
                opt_eval = torch.compile(multiset_eval, mode="max-autotune")
                print("Eval compiled")
                for i in range(5):
                    s = time.time()
                    opt_eval(opt_net, data_iters)
                    e = time.time()
                    print("Eval time {}: {}".format(i, (e-s)/sum([len(d) for d in data_iters])))
                s = time.time()
                opt_eval(opt_net, data_iters)
                e = time.time()
                print("Eval time final: {}".format((e-s)/sum([len(d) for d in data_iters])))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("modelfile", help="file containing Pytorch model", type=str)
    parser.add_argument("--out", help="file to write Torchscript model", type=str)
    parser.add_argument("--dir", help="Directory containing data instead of file", action="store_true", default=False)
    parser.add_argument("--prefix", help="Prefix for combined models", type=str, default="")
    parser.add_argument("--miss_file", help="File to get exact number of classes (onehot model only)", default=None, type=str)
    parser.add_argument('--max_classes', help='classes in embedding layer', type=int, default=20000)
    parser.add_argument("--eh", help="Embedding and hidden dimensions", type=int, default=32)
    parser.add_argument("--compile", help="Compile model for timing", action="store_true", default=False)
    parser.add_argument("--train", help="time model training", action="store_true", default=False)
    parser.add_argument("--eval", help="time model inference", action="store_true", default=False)
    parser.add_argument("-b", help="batch size for training", type=int, default=32)
    parser.add_argument("-q", help="quantize model", action="store_true", default=False)

    args = parser.parse_args()
    main(args)