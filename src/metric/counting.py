import torch
from torchmetrics import Metric
import numpy as np

class CountingMetric(Metric):
    def __init__(self, classes, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        super(CountingMetric, self).__init__()
        self.classes = classes
       
        self.add_state('mae_per_image', default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state('mae_per_image_norm', default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, inputs: torch.Tensor, targets: torch.Tensor):
        
        for input_b, target_b in zip(inputs, targets):
            mae_per_class = 0.0
            targets_per_class = 0.0
            
            for cl in np.arange(self.classes):
                mae_per_class += abs(len(input_b[input_b[:, 0]==cl]) - len(target_b[target_b[:, 0]==cl]))
                targets_per_class += len(target_b[target_b[:, 0]==cl])
                
            self.mae_per_image += mae_per_class / self.classes
            self.mae_per_image_norm += torch.div(mae_per_class / self.classes, targets_per_class+1)
            
            self.count += 1

    def compute(self):
        return self.mae_per_image / self.count, self.mae_per_image_norm / self.count

class CountingMAEMetric(Metric):
    def __init__(self, classes, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        super(CountingMAEMetric, self).__init__()
        self.classes = classes

        self.add_state('mae_per_image', default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state('mae_per_image_norm', default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, inputs: torch.Tensor, targets: torch.Tensor):
        
        for input_b, target_b in zip(inputs, targets):
            mae_per_class = 0.0
            targets_per_class = 0.0
            
            for cl in np.arange(self.classes):
                mae_per_class += abs(torch.div(torch.sum(input_b[cl]), 100) - len(target_b[target_b[:, 0]==cl]))
                targets_per_class += len(target_b[target_b[:, 0]==cl])
                
            self.mae_per_image += mae_per_class / self.classes
            self.mae_per_image_norm += torch.div(mae_per_class / self.classes, targets_per_class+1)
            self.count += 1

    def compute(self):
        return self.mae_per_image / self.count, self.mae_per_image_norm / self.count
