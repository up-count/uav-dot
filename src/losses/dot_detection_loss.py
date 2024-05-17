# partly based on https://github.com/WongKinYiu/yolov7/blob/84932d70fb9e2932d0a70e4a1f02a1d6dd1dd6ca/utils/metrics.py#L136 
import numpy as np
import torch

from src.metric.detection_f1 import DetectionF1Metric


class DotDetectionLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, pred, gt):
        obj_loss, reg_loss = torch.tensor(0.0), torch.tensor(0.0)

        for input_b, target_b in zip(pred, gt):
            target_classes = target_b[:, 0].cpu()
            targets = target_b[:, 1:3].cpu()

            classes = input_b[:, 0].cpu()
            centers = input_b[:, 1:3].cpu()
            confs = input_b[:, 3].cpu()

            correct = torch.zeros(
                (classes.shape[0],), dtype=torch.bool)
            reg = torch.zeros((classes.shape[0],), dtype=torch.float32)


            if centers.shape[0] > 0 and targets.shape[0] > 0:
                scores = DetectionF1Metric.radial_score(
                    centers, targets, 2.5, sigmoid=True)

                tc = target_classes.unsqueeze(1).repeat(1, scores.shape[0])
                correct_class = (tc == classes).T

                assert scores.shape == correct_class.shape, f'{scores.shape} != {correct_class.shape}'

                x = torch.where(
                    (scores >= DetectionF1Metric.correct_threshold) & correct_class)

                if x[0].shape[0]:
                    matches = torch.cat(
                        (torch.stack(x, 1), scores[x[0], x[1]][:, None]), 1).detach().cpu().numpy()
                    if x[0].shape[0] > 1:
                        matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

                    correct[matches[:, 0].astype(int)] = True

                reg[correct] = scores[correct].max(1)[0]

                obj_loss += self.bce(confs, correct.float())
                reg_loss += (1.0 - reg).mean()
            else:
                obj_loss += 1.0
                reg_loss += 1.0
                
        obj_loss /= len(pred)
        reg_loss /= len(pred)

        obj_loss = obj_loss.to(pred[0].device)
        reg_loss = reg_loss.to(pred[0].device)

        return obj_loss, reg_loss, {'obj_loss': obj_loss, 'reg_loss': reg_loss}
