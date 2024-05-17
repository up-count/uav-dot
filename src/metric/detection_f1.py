import torch
from torchmetrics import Metric
import numpy as np

class DetectionF1Metric(Metric):
    correct_threshold = 0.5

    def __init__(self, dist_sync_on_step: bool = False, correct_distance: float = 5.0, return_correct: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        super(DetectionF1Metric, self).__init__()
        self.correct_distance = correct_distance
        self.return_correct = return_correct

        self.add_state('tp', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('fp', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('fn', default=torch.tensor(0), dist_reduce_fx="sum")
        
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx="sum")

        if self.return_correct:
            self.correct = None


    def update(self, inputs: torch.Tensor, targets: torch.Tensor):
        
        for input_b, target_b in zip(inputs, targets):
            target_classes = target_b[:, 0]
            targets = target_b[:, 1:3]
            
            classes = input_b[:, 0]
            centers = input_b[:, 1:3]
                
            correct = torch.zeros((classes.shape[0],), dtype=torch.bool).to(classes.device)
                
            if centers.shape[0] > 0:
                scores = self.radial_score(centers, targets, self.correct_distance)
                
                tc = target_classes.unsqueeze(1).repeat(1, scores.shape[0])
                correct_class = (tc == classes).T
                
                assert scores.shape == correct_class.shape, f'{scores.shape} != {correct_class.shape}'
                
                x = torch.where((scores >= self.correct_threshold) & correct_class)
                
                if x[0].shape[0]:
                    matches = torch.cat((torch.stack(x, 1), scores[x[0], x[1]][:, None]), 1).cpu().numpy()
                    if x[0].shape[0] > 1:
                        matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                        
                    correct[matches[:, 0].astype(int)] = True
            
            TP = correct.sum()
            FP = correct.shape[0] - TP
            FN = target_classes.shape[0] - TP
                        
            self.tp += TP
            self.fp += FP
            self.fn += FN
            
            self.count += 1
            
            if self.return_correct:
                self.correct = correct

    def compute(self):
        precission = (self.tp + 1e-6) / (self.tp + 1e-6 + self.fp + 1e-6)
        recall = (self.tp + 1e-6) / (self.tp + self.fn + 1e-6)
        f1 = (2 * precission * recall) / (precission + recall)
        
        if self.return_correct:
            return f1, precission, recall, self.correct
        else:
            return f1, precission, recall

    @staticmethod
    def radial_score(pred, target, thresh, sigmoid=False):
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
        
        dist = torch.cdist(pred.float(), target.float())[0]
        
        if sigmoid:
            inv_dist = (thresh - dist)
            return torch.sigmoid(inv_dist)
        else:
            return dist <= thresh
