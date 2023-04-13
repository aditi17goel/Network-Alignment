from input.dataset import Dataset
from time import time
from algorithms import *
from evaluation.metrics import get_statistics
import utils.graph_utils as graph_utils
import random
import numpy as np
import torch
import argparse
import os
import pdb
from utils.graph_utils import load_gt
import torch.nn.functional as F
# import timesd

def parse_args():
    parser = argparse.ArgumentParser(description="Network alignment")
    parser.add_argument('--source_dataset', default="dataspace/douban/online/graphsage/")
    parser.add_argument('--target_dataset', default="dataspace/douban/offline/graphsage/")
    parser.add_argument('--groundtruth',    default="dataspace/douban/dictionaries/groundtruth")
    parser.add_argument('--seed',           default=123,    type=int)
    subparsers = parser.add_subparsers(dest="algorithm", help='Choose 1 of the algorithm from: IsoRank, FINAL, UniAlign, NAWAL, DeepLink, REGAL, IONE, PALE')    


    # GAlign
    parser_GAlign = subparsers.add_parser("GAlign", help="GAlign algorithm")
    parser_GAlign.add_argument('--cuda',                action="store_true")
    parser_GAlign.add_argument('--embedding_dim',       default=200,         type=int)
    parser_GAlign.add_argument('--GAlign_epochs',    default=20,        type=int)
    parser_GAlign.add_argument('--lr', default=0.01, type=float)
    parser_GAlign.add_argument('--num_GCN_blocks', type=int, default=2)
    parser_GAlign.add_argument('--act', type=str, default='tanh')
    parser_GAlign.add_argument('--log', action="store_true", help="Just to print loss")
    parser_GAlign.add_argument('--invest', action="store_true", help="To do some statistics")
    parser_GAlign.add_argument('--input_dim', default=100, help="Just ignore it")
    parser_GAlign.add_argument('--alpha0', type=float, default=1)
    parser_GAlign.add_argument('--alpha1', type=float, default=1)
    parser_GAlign.add_argument('--alpha2', type=float, default=1)
    parser_GAlign.add_argument('--noise_level', type=float, default=0.01)

    # refinement
    parser_GAlign.add_argument('--refinement_epochs', default=10, type=int)
    parser_GAlign.add_argument('--threshold_refine', type=float, default=0.94, help="The threshold value to get stable candidates")

    # loss
    parser_GAlign.add_argument('--beta', type=float, default=0.8, help='balancing source-target and source-augment')
    parser_GAlign.add_argument('--threshold', type=float, default=0.01, help='confidence threshold for adaptivity loss')
    parser_GAlign.add_argument('--coe_consistency', type=float, default=0.8, help='balancing consistency and adaptivity loss')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    start_time = time()
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)

    source_dataset = Dataset(args.source_dataset)
    target_dataset = Dataset(args.target_dataset)
    groundtruth = graph_utils.load_gt(args.groundtruth, source_dataset.id2idx, target_dataset.id2idx, 'dict')

    algorithm = args.algorithm

    if algorithm == "IsoRank":
        train_dict = None
        if args.train_dict != "":
            train_dict = graph_utils.load_gt(args.train_dict, source_dataset.id2idx, target_dataset.id2idx, 'dict')
        model = IsoRank(source_dataset, target_dataset, args.H, args.alpha, args.max_iter, args.tol, train_dict=train_dict)
    elif algorithm == "FINAL":
        train_dict = None
        if args.train_dict != "":
            train_dict = graph_utils.load_gt(args.train_dict, source_dataset.id2idx, target_dataset.id2idx, 'dict')
        model = FINAL(source_dataset, target_dataset, H=args.H, alpha=args.alpha, maxiter=args.max_iter, tol=args.tol, train_dict=train_dict)
    elif algorithm == "REGAL":
        model = REGAL(source_dataset, target_dataset, max_layer=args.max_layer, alpha=args.alpha, k=args.k, num_buckets=args.buckets,
                      gammastruc = args.gammastruc, gammaattr = args.gammaattr, normalize=True, num_top=args.num_top)
    elif algorithm == "BigAlign":
        model = BigAlign(source_dataset, target_dataset, lamb=args.lamb)
    elif algorithm == "IONE":
        model = IONE(source_dataset, target_dataset, gt_train=args.train_dict, epochs=args.epochs, dim=args.dim, seed=args.seed, learning_rate=args.lr)
    elif algorithm == "DeepLink":
        model = DeepLink(source_dataset, target_dataset, args)
    elif algorithm == "GAlign":
        model = GAlign(source_dataset, target_dataset, args)
    elif algorithm == "PALE":
        model = PALE(source_dataset, target_dataset, args)
    elif algorithm == "CENALP":
        model = CENALP(source_dataset, target_dataset, args)
    elif algorithm == "NAWAL":
        model = NAWAL(source_dataset, target_dataset, args)
    else:
        raise Exception("Unsupported algorithm")


    S = model.align()
    print("-"*100)
    acc, MAP, top5, top10 = get_statistics(S, groundtruth, use_greedy_match=False, get_all_metric=True)
    print("Accuracy: {:.4f}".format(acc))
    print("MAP: {:.4f}".format(MAP))
    print("Precision_5: {:.4f}".format(top5))
    print("Precision_10: {:.4f}".format(top10))
    print("-"*100)
    print('Running time: {}'.format(time()-start_time))
