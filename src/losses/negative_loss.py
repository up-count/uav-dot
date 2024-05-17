# based on https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/losses.py#L42C8-L42C8

import torch
import torch.nn as nn

class NegativeLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):

        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()
        
        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        pos_loss = torch.log(pred + 1e-6) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred + 1e-6) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
                       
        return loss
