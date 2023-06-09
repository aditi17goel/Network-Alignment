from algorithms.network_alignment_model import NetworkAlignmentModel
from evaluation.metrics import get_statistics
from algorithms.GAlign.embedding_model import G_Align as Multi_Order, StableFactor
from input.dataset import Dataset
from utils.graph_utils import load_gt
import torch.nn.functional as F
import torch.nn as nn
from algorithms.GAlign.utils import *
from algorithms.GAlign.losses import *

import torch
import numpy as np
import networkx as nx
import random 
import numpy as np

import argparse
import os
import time
import sys

from torch.autograd import Variable
from tqdm import tqdm


class GAlign(NetworkAlignmentModel):
    """
    GAlign model for networks alignment task
    """
    def __init__(self, source_dataset, target_dataset, args):
        """
        :params source_dataset: source graph
        :params target_dataset: target graph
        :params args: more config params
        """
        super(GAlign, self).__init__(source_dataset, target_dataset)
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.alphas = [args.alpha0, args.alpha1, args.alpha2]
        self.args = args
        self.full_dict = load_gt(args.groundtruth, source_dataset.id2idx, target_dataset.id2idx, 'dict')


    def graph_augmentation(self, dataset, type_aug='remove_edges'):
        """
        Generate small noisy graph from original graph
        :params dataset: original graph
        :params type_aug: type of noise added for generating new graph
        """
        edges = dataset.get_edges()
        adj = dataset.get_adjacency_matrix()
        
        if type_aug == "remove_edges":
            num_edges = len(edges)
            num_remove = int(len(edges) * self.args.noise_level)
            index_to_remove = np.random.choice(np.arange(num_edges), num_remove, replace=False)
            edges_to_remove = edges[index_to_remove]
            for i in range(len(edges_to_remove)):
                adj[edges_to_remove[i, 0], edges_to_remove[i, 1]] = 0
                adj[edges_to_remove[i, 1], edges_to_remove[i, 0]] = 0
        elif type_aug == "add_edges":
            num_edges = len(edges)
            num_add = int(len(edges) * self.args.noise_level)
            count_add = 0
            while count_add < num_add:
                random_index = np.random.randint(0, adj.shape[1], 2)
                if adj[random_index[0], random_index[1]] == 0:
                    adj[random_index[0], random_index[1]] = 1
                    adj[random_index[1], random_index[0]] = 1
                    count_add += 1
        elif type_aug == "change_feats":
            feats = np.copy(dataset.features)
            num_nodes = adj.shape[0]
            num_nodes_change_feats = int(num_nodes * self.args.noise_level)
            node_to_change_feats = np.random.choice(np.arange(0, adj.shape[0]), num_nodes_change_feats, replace=False)
            for node in node_to_change_feats:
                feat_node = feats[node]
                feat_node[feat_node == 1] = 0
                feat_node[np.random.randint(0, feats.shape[1], 1)[0]] = 1
            feats = torch.FloatTensor(feats)
            if self.args.cuda:
                feats = feats.cuda()
            return feats
        new_adj_H, _ = Laplacian_graph(adj)
        if self.args.cuda:
            new_adj_H = new_adj_H.cuda()
        return new_adj_H




    def align(self):
        """
        The main function of GAlign
        """
        source_A_hat, target_A_hat, source_feats, target_feats = self.get_elements()
        print("Running Multi-level embedding")
        GAlign = self.multi_level_embed(source_A_hat, target_A_hat, source_feats, target_feats)
        source_A_hat = source_A_hat.to_dense()
        target_A_hat = target_A_hat.to_dense()
        GAlign_S = self.refine(GAlign, source_A_hat, target_A_hat, 0.94)
        return GAlign_S


    def multi_level_embed(self, source_A_hat, target_A_hat, source_feats, target_feats):
        """
        Input: SourceGraph and TargetGraph
        Output: Embedding of those graphs using Multi_order_embedding model
        """
        GAlign = Multi_Order(
            activate_function = self.args.act,
            num_GCN_blocks = self.args.num_GCN_blocks,
            input_dim = self.args.input_dim,
            output_dim = self.args.embedding_dim,
            num_source_nodes = len(source_A_hat),
            num_target_nodes = len(target_A_hat),
            source_feats = source_feats,
            target_feats = target_feats
        )

        if self.args.cuda:
            GAlign = GAlign.cuda()

        structural_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, GAlign.parameters()), lr=self.args.lr)

        new_source_A_hats = []
        new_target_A_hats = []
        new_source_A_hats.append(self.graph_augmentation(self.source_dataset, 'remove_edgse'))
        new_source_A_hats.append(self.graph_augmentation(self.source_dataset, 'add_edges'))
        new_source_A_hats.append(source_A_hat)
        new_source_feats = self.graph_augmentation(self.source_dataset, 'change_feats')
        new_target_A_hats.append(self.graph_augmentation(self.target_dataset, 'remove_edgse'))
        new_target_A_hats.append(self.graph_augmentation(self.target_dataset, 'add_edges'))
        new_target_A_hats.append(target_A_hat)
        new_target_feats = self.graph_augmentation(self.target_dataset, 'change_feats')

        for epoch in range(self.args.GAlign_epochs):
            if self.args.log:
                print("Structure learning epoch: {}".format(epoch))
            for i in range(2):
                for j in range(len(new_source_A_hats)):
                    structural_optimizer.zero_grad()
                    if i == 0:
                        A_hat = source_A_hat
                        augment_A_hat = new_source_A_hats[j]
                        outputs = GAlign(source_A_hat, 's')
                        if j < 3:
                            augment_outputs = GAlign(augment_A_hat, 's')
                        else:
                            augment_outputs = GAlign(augment_A_hat, 's', new_source_feats)
                    else:
                        A_hat = target_A_hat
                        augment_A_hat = new_target_A_hats[j]
                        outputs = GAlign(target_A_hat, 't')
                        if j < 3:
                            augment_outputs = GAlign(augment_A_hat, 't')
                        else:
                            augment_outputs = GAlign(augment_A_hat, 't', new_target_feats)
                    consistency_loss = self.linkpred_loss(outputs[-1], A_hat)
                    augment_consistency_loss = self.linkpred_loss(augment_outputs[-1], augment_A_hat)
                    consistency_loss = self.args.beta * consistency_loss + (1-self.args.beta) * augment_consistency_loss
                    diff = torch.abs(outputs[-1] - augment_outputs[-1])
                    noise_adaptivity_loss = (diff[diff < self.args.threshold] ** 2).sum() / len(outputs)
                    loss = self.args.coe_consistency * consistency_loss + (1 - self.args.coe_consistency) * noise_adaptivity_loss
                    if self.args.log:
                        print("Loss: {:.4f}".format(loss.data))
                    loss.backward()
                    structural_optimizer.step()
        GAlign.eval()
        return GAlign





    def get_elements(self):
        """
        Compute Normalized Laplacian matrix
        Preprocessing nodes attribute
        """
        source_A_hat, _ = Laplacian_graph(self.source_dataset.get_adjacency_matrix())
        target_A_hat, _ = Laplacian_graph(self.target_dataset.get_adjacency_matrix())
        if self.args.cuda:
            source_A_hat = source_A_hat.cuda()
            target_A_hat = target_A_hat.cuda()

        source_feats = self.source_dataset.features
        target_feats = self.target_dataset.features

        if source_feats is None:
            source_feats = np.zeros((len(self.source_dataset.G.nodes()), 1))
            target_feats = np.zeros((len(self.target_dataset.G.nodes()), 1))
        
        for i in range(len(source_feats)):
            if source_feats[i].sum() == 0:
                source_feats[i, -1] = 1
        for i in range(len(target_feats)):
            if target_feats[i].sum() == 0:
                target_feats[i, -1] = 1
        if source_feats is not None:
            source_feats = torch.FloatTensor(source_feats)
            target_feats = torch.FloatTensor(target_feats)
            if self.args.cuda:
                source_feats = source_feats.cuda()
                target_feats = target_feats.cuda()
        source_feats = F.normalize(source_feats)
        target_feats = F.normalize(target_feats)
        return source_A_hat, target_A_hat, source_feats, target_feats


    def linkpred_loss(self, embedding, A):
        pred_adj = torch.matmul(F.normalize(embedding), F.normalize(embedding).t())
        if self.args.cuda:
            pred_adj = F.normalize((torch.min(pred_adj, torch.Tensor([1]).cuda())), dim = 1)
        else:
            pred_adj = F.normalize((torch.min(pred_adj, torch.Tensor([1]))), dim = 1)
        #linkpred_losss = (pred_adj - A[index]) ** 2
        linkpred_losss = (pred_adj - A) ** 2
        linkpred_losss = linkpred_losss.sum() / A.shape[1]
        return linkpred_losss


    def refine(self, GAlign, source_A_hat, target_A_hat, threshold):
        refinement_model = StableFactor(len(source_A_hat), len(target_A_hat), self.args.cuda)
        if self.args.cuda: 
            refinement_model = refinement_model.cuda()
        S_max = None
        source_outputs = GAlign(refinement_model(source_A_hat, 's'), 's')
        target_outputs = GAlign(refinement_model(target_A_hat, 't'), 't')
        acc, S = get_acc(source_outputs, target_outputs, self.full_dict, self.alphas, just_S=True)
        score = np.max(S, axis=1).mean()
        self.GAlign_S = S
        return self.GAlign_S


    def log_and_evaluate(self, embedding_model, refinement_model, source_A_hat, target_A_hat):
        embedding_model.eval()
        source_outputs = embedding_model(refinement_model(source_A_hat, 's'), 's')
        target_outputs = embedding_model(refinement_model(target_A_hat, 't'), 't')
        print("-"* 100)
        log, self.S = get_acc(source_outputs, target_outputs, self.full_dict, self.alphas)
        print(self.alphas)
        print(log)
        return source_outputs, target_outputs
