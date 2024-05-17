import torch

from src.losses.negative_loss import NegativeLoss
from src.losses.dot_detection_loss import DotDetectionLoss



class DotLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.ng_loss = NegativeLoss()
        self.reg_loss = DotDetectionLoss()
        
    def forward(self, pred_mask, pred_points, gt_mask, gt_points, is_stage=False):
        
        n_loss = self.ng_loss(torch.sigmoid(pred_mask), gt_mask)
        losses_dict = {'ng_loss': n_loss}
        
        if is_stage:
            return n_loss, losses_dict
        
        obj_loss, reg_loss, r_dict = self.reg_loss(pred_points, gt_points)
                       
        losses_dict.update(r_dict)
        
        total_loss = \
            0.25 * n_loss + \
            1.0 * obj_loss + \
            2.0 * reg_loss
                               
        return total_loss, losses_dict
