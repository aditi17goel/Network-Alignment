import numpy as np
import torch

        
def mapping_loss(source, target, gt_train):
    source_train_nodes = torch.LongTensor(np.array(list(gt_train.keys()))).cuda()
    target_train_nodes = torch.LongTensor(np.array(list(gt_train.values()))).cuda()
    losses = [torch.sum((source[i][source_train_nodes] - \
        target[i][target_train_nodes])**2) / len(source_train_nodes) for i in range(len(source))]
    return losses


def hinge_loss(source_outputs, target_outputs, neg_source_indices, neg_target_indices, gt_train, neg_sample_size):
    source_train_nodes = torch.LongTensor(np.array(list(gt_train.keys()))).cuda()
    target_train_nodes = torch.LongTensor(np.array(list(gt_train.values()))).cuda()

    losses = []
    for i in range(len(source_outputs)):
        source_train_emb = source_outputs[i][source_train_nodes]
        target_train_emb = target_outputs[i][target_train_nodes]
        loss_i = torch.zeros(len(source_train_emb)).cuda()
        anchor_simi = torch.sum(source_train_emb * target_train_emb, dim=1)
        
        for j in range(neg_sample_size):
            neg_source_index_source = neg_source_indices[j][0]
            neg_source_index_target = neg_source_indices[j][1]
            neg_target_index_source = neg_target_indices[j][0]
            neg_target_index_target = neg_target_indices[j][1]

            neg_source_emb_source = source_outputs[i][neg_source_index_source]
            neg_source_emb_target = source_outputs[i][neg_source_index_target]
            neg_target_emb_source = target_outputs[i][neg_target_index_source]
            neg_target_emb_target = target_outputs[i][neg_target_index_target]

            neg_source_simi_source = torch.sum(source_train_emb * neg_source_emb_source, dim=1)
            neg_source_simi_target = torch.sum(source_train_emb * neg_target_emb_source, dim=1)
            neg_target_simi_target = torch.sum(target_train_emb * neg_target_emb_target, dim=1)
            neg_target_simi_source = torch.sum(target_train_emb * neg_source_emb_target, dim=1)
            A = 1 - anchor_simi
            A = 2

            loss_j_1 = neg_source_simi_source + A
            loss_j_1[loss_j_1 < 0] = 0
            loss_j_2 = neg_source_simi_target + A
            loss_j_2[loss_j_2 < 0] = 0
            loss_j_3 = neg_target_simi_target + A
            loss_j_3[loss_j_3 < 0] = 0
            loss_j_4 = neg_target_simi_source + A
            loss_j_4[loss_j_4 < 0] = 0
            loss_i += loss_j_1 + loss_j_2 + loss_j_3 + loss_j_4
        
        loss_i = loss_i/ (len(source_train_nodes) * 5 * 4)
        loss_i = loss_i.sum()
        losses.append(loss_i)
    return losses

        
def simple_loss(source_outputs, target_outputs, neg_source_indices, neg_target_indices, neg_sample_size):
    losses = []
    for i in range(len(source_outputs)):
        loss_i = torch.zeros(100).cuda()
        
        for j in range(neg_sample_size):
            neg_source_index_1 = neg_source_indices[j][0]
            neg_source_index_2 = neg_source_indices[j][1]
            neg_target_index_1 = neg_target_indices[j][0]
            neg_target_index_2 = neg_target_indices[j][1]

            neg_source_emb_1 = source_outputs[i][neg_source_index_1]
            neg_source_emb_2 = source_outputs[i][neg_source_index_2]
            neg_target_emb_1 = target_outputs[i][neg_target_index_1]
            neg_target_emb_2 = target_outputs[i][neg_target_index_2]

            neg_source_simi_source = torch.sum(neg_source_emb_1 * neg_source_emb_2, dim=1)
            # neg_source_simi_target = torch.sum(neg_source_emb_1 * neg_target_emb_2, dim=1)
            # neg_target_simi_source = torch.sum(neg_target_emb_1 * neg_source_emb_2, dim=1)
            neg_target_simi_target = torch.sum(neg_target_emb_1 * neg_target_emb_2, dim=1)

            A = 2

            loss_j_1 = neg_source_simi_source + A
            loss_j_1[loss_j_1 < 0] = 0
            # loss_j_2 = neg_source_simi_target + A
            # loss_j_2[loss_j_2 < 0] = 0
            loss_j_3 = neg_target_simi_target + A
            loss_j_3[loss_j_3 < 0] = 0
            # loss_j_4 = neg_target_simi_source + A
            # loss_j_4[loss_j_4 < 0] = 0
            #loss_i += loss_j_1 + loss_j_2 + loss_j_3 + loss_j_4
            loss_i += loss_j_1 + loss_j_3
        
        loss_i = loss_i/ (100 * 5 * 4)
        loss_i = loss_i.sum()
        losses.append(loss_i)
    return losses
