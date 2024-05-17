import numpy as np
import torch

from src.metric.detection_f1 import DetectionF1Metric


class RegressionLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.bce = torch.nn.BCELoss()

    def forward(self, pred, gt):
        tp, fp, fn = 0.0, 0.0, 0.0
          
        for input_b, target_b in zip(pred, gt):
            target_classes = target_b[:, 0]
            targets = target_b[:, 1:3]
            
            classes = input_b[:, 0]
            centers = input_b[:, 1:3]
                
            correct = torch.zeros((classes.shape[0],), dtype=torch.bool).to(classes.device)
                
            if centers.shape[0] > 0:
                scores = DetectionF1Metric.radial_score(centers, targets, 5.0, sigmoid=True)
                
                tc = target_classes.unsqueeze(1).repeat(1, scores.shape[0])
                correct_class = (tc == classes).T
                
                assert scores.shape == correct_class.shape, f'{scores.shape} != {correct_class.shape}'
                
                x = torch.where((scores >= DetectionF1Metric.correct_threshold) & correct_class)
                
                if x[0].shape[0]:
                    matches = torch.cat((torch.stack(x, 1), scores[x[0], x[1]][:, None]), 1).detach().cpu().numpy()
                    if x[0].shape[0] > 1:
                        matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                        
                    correct[matches[:, 0].astype(int)] = True
                    
            TP = correct.sum()
            FP = correct.shape[0] - TP
            FN = target_classes.shape[0] - TP
                        
            tp += TP
            fp += FP
            fn += FN
        
        precission = (tp + 1e-6) / (tp + 1e-6 + fp + 1e-6)
        recall = (tp + 1e-6) / (tp + fn + 1e-6)
        f1 = (2 * precission * recall) / (precission + recall)           
                       
        return 1.0 - f1
