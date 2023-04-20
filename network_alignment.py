from input.dataset import Dataset
from time import time
from algorithms import *
from sklearn.svm import LinearSVC, SVC
from evaluation.metrics import get_statistics
from adgcl.unsupervised.encoder import TUEncoder
from adgcl.unsupervised.learning import GInfoMinMax
import utils.graph_utils as graph_utils
import random
import numpy as np
import torch
import argparse
import os
import pdb
import torch_geometric as tg
from utils.graph_utils import load_gt
import torch.nn.functional as F
from adgcl.datasets import TUDataset, TUEvaluator
from adgcl.unsupervised.view_learner import ViewLearner
from adgcl.unsupervised.embedding_evaluation import EmbeddingEvaluation

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
#    target_dataset = Dataset(args.target_dataset)
    print(source_dataset)
    dataloader = tg.loader.DataLoader(source_dataset, batch_size=32, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GInfoMinMax(
        TUEncoder(num_dataset_features=1, emb_dim=32, num_gc_layers=5, drop_ratio=0.0, pooling_type='standard'),
        32).to(device)
    
    model_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    evaluator = TUEvaluator()

    view_learner = ViewLearner(TUEncoder(num_dataset_features=1, emb_dim=32, num_gc_layers=5, drop_ratio=0.0, pooling_type='standard'),
                               mlp_edge_model_dim=64).to(device)
    view_optimizer = torch.optim.Adam(view_learner.parameters(), lr=0.001)
    ee = EmbeddingEvaluation(LinearSVC(dual=False, fit_intercept=True), evaluator, 'classification', 1,
                             device, param_search=True)
    train_score, val_score, test_score = ee.kf_embedding_evaluation(model.encoder, dataloader)
    logging.info(
        "Before training Embedding Eval Scores: Train: {} Val: {} Test: {}".format(train_score, val_score,
                                                                                         test_score))


    model_losses = []
    view_losses = []
    view_regs = []
    valid_curve = []
    test_curve = []
    train_curve = []

    for epoch in range(1, args.epochs + 1):
        model_loss_all = 0
        view_loss_all = 0
        reg_all = 0
        for batch in dataloader:
            # set up
            batch = batch.to(device)

            # train view to maximize contrastive loss
            view_learner.train()
            view_learner.zero_grad()
            model.eval()

            x, _ = model(batch.batch, batch.x, batch.edge_index, None, None)

            edge_logits = view_learner(batch.batch, batch.x, batch.edge_index, None)

            temperature = 1.0
            bias = 0.0 + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = gate_inputs.to(device)
            gate_inputs = (gate_inputs + edge_logits) / temperature
            batch_aug_edge_weight = torch.sigmoid(gate_inputs).squeeze()

            x_aug, _ = model(batch.batch, batch.x, batch.edge_index, None, batch_aug_edge_weight)

            # regularization

            row, col = batch.edge_index
            edge_batch = batch.batch[row]
            edge_drop_out_prob = 1 - batch_aug_edge_weight

            uni, edge_batch_num = edge_batch.unique(return_counts=True)
            sum_pe = scatter(edge_drop_out_prob, edge_batch, reduce="sum")

            reg = []
            for b_id in range(args.batch_size):
                if b_id in uni:
                    num_edges = edge_batch_num[uni.tolist().index(b_id)]
                    reg.append(sum_pe[b_id] / num_edges)
                else:
                    # means no edges in that graph. So don't include.
                    pass
            num_graph_with_edges = len(reg)
            reg = torch.stack(reg)
            reg = reg.mean()


            view_loss = model.calc_loss(x, x_aug) - (args.reg_lambda * reg)
            view_loss_all += view_loss.item() * batch.num_graphs
            reg_all += reg.item()
            # gradient ascent formulation
            (-view_loss).backward()
            view_optimizer.step()


            # train (model) to minimize contrastive loss
            model.train()
            view_learner.eval()
            model.zero_grad()

            x, _ = model(batch.batch, batch.x, batch.edge_index, None, None)
            edge_logits = view_learner(batch.batch, batch.x, batch.edge_index, None)

            temperature = 1.0
            bias = 0.0 + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = gate_inputs.to(device)
            gate_inputs = (gate_inputs + edge_logits) / temperature
            batch_aug_edge_weight = torch.sigmoid(gate_inputs).squeeze().detach()

            x_aug, _ = model(batch.batch, batch.x, batch.edge_index, None, batch_aug_edge_weight)

            model_loss = model.calc_loss(x, x_aug)
            model_loss_all += model_loss.item() * batch.num_graphs
            # standard gradient descent formulation
            model_loss.backward()
            model_optimizer.step()

        fin_model_loss = model_loss_all / len(dataloader)
        fin_view_loss = view_loss_all / len(dataloader)
        fin_reg = reg_all / len(dataloader)

        logging.info('Epoch {}, Model Loss {}, View Loss {}, Reg {}'.format(epoch, fin_model_loss, fin_view_loss, fin_reg))
        model_losses.append(fin_model_loss)
        view_losses.append(fin_view_loss)
        view_regs.append(fin_reg)
        if epoch % args.eval_interval == 0:
            model.eval()

            train_score, val_score, test_score = ee.kf_embedding_evaluation(model.encoder, dataset)

            logging.info(
                "Metric: {} Train: {} Val: {} Test: {}".format(evaluator.eval_metric, train_score, val_score, test_score))

            train_curve.append(train_score)
            valid_curve.append(val_score)
            test_curve.append(test_score)

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    logging.info('FinishedTraining!')
    logging.info('BestEpoch: {}'.format(best_val_epoch))
    logging.info('BestTrainScore: {}'.format(best_train))
    logging.info('BestValidationScore: {}'.format(valid_curve[best_val_epoch]))
    logging.info('FinalTestScore: {}'.format(test_curve[best_val_epoch]))


    print("hi",valid_curve[best_val_epoch])
    print("hii",test_curve[best_val_epoch])

#    groundtruth = graph_utils.load_gt(args.groundtruth, source_dataset.id2idx, target_dataset.id2idx, 'dict')
#
#    algorithm = args.algorithm
#
#    if algorithm == "IsoRank":
#        train_dict = None
#        if args.train_dict != "":
#            train_dict = graph_utils.load_gt(args.train_dict, source_dataset.id2idx, target_dataset.id2idx, 'dict')
#        model = IsoRank(source_dataset, target_dataset, args.H, args.alpha, args.max_iter, args.tol, train_dict=train_dict)
#    elif algorithm == "FINAL":
#        train_dict = None
#        if args.train_dict != "":
#            train_dict = graph_utils.load_gt(args.train_dict, source_dataset.id2idx, target_dataset.id2idx, 'dict')
#        model = FINAL(source_dataset, target_dataset, H=args.H, alpha=args.alpha, maxiter=args.max_iter, tol=args.tol, train_dict=train_dict)
#    elif algorithm == "REGAL":
#        model = REGAL(source_dataset, target_dataset, max_layer=args.max_layer, alpha=args.alpha, k=args.k, num_buckets=args.buckets,
#                      gammastruc = args.gammastruc, gammaattr = args.gammaattr, normalize=True, num_top=args.num_top)
#    elif algorithm == "BigAlign":
#        model = BigAlign(source_dataset, target_dataset, lamb=args.lamb)
#    elif algorithm == "IONE":
#        model = IONE(source_dataset, target_dataset, gt_train=args.train_dict, epochs=args.epochs, dim=args.dim, seed=args.seed, learning_rate=args.lr)
#    elif algorithm == "DeepLink":
#        model = DeepLink(source_dataset, target_dataset, args)
#    elif algorithm == "GAlign":
#        model = GAlign(source_dataset, target_dataset, args)
#    elif algorithm == "PALE":
#        model = PALE(source_dataset, target_dataset, args)
#    elif algorithm == "CENALP":
#        model = CENALP(source_dataset, target_dataset, args)
#    elif algorithm == "NAWAL":
#        model = NAWAL(source_dataset, target_dataset, args)
#    else:
#        raise Exception("Unsupported algorithm")
#
#
#    S = model.align()
#    print("-"*100)
#    acc, MAP, top5, top10 = get_statistics(S, groundtruth, use_greedy_match=False, get_all_metric=True)
#    print("Accuracy: {:.4f}".format(acc))
#    print("MAP: {:.4f}".format(MAP))
#    print("Precision_5: {:.4f}".format(top5))
#    print("Precision_10: {:.4f}".format(top10))
#    print("-"*100)
#    print('Running time: {}'.format(time()-start_time))
