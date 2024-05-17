from typing import Optional

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.nn.functional

from src.metric.detection_f1 import DetectionF1Metric
from src.losses.dot_loss import DotLoss
from src.metric.counting import CountingMetric
from src.model.dot_model import DotModel


class DotRegressor(pl.LightningModule):
    def __init__(self,
                 encoder_name: str,
                 input_channels: int,
                 output_channels: int,
                 spatial_mode: str,
                 loss_function: str,
                 lr: float,
                 train_steps: int,
                 visualize_test_images: bool,
                 obj_threshold: float,
                 image_size: tuple[int, int],
                 mask_size: tuple[int, int],
                 ):
        super().__init__()

        # params
        self._encoder_name = encoder_name
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._spatial_mode = spatial_mode
        self._loss_function = loss_function
        self._lr = lr
        self._train_steps = train_steps
        self._visualize_test_images = visualize_test_images
        self._obj_threshold = obj_threshold
        self._image_size = image_size
        self._mask_size = mask_size

        # network
        self.network = DotModel(
            encoder_name=self._encoder_name,
            classes=self._output_channels,
            reduce_spatial_mode=self._spatial_mode,
            image_size=self._image_size,
            mask_size=self._mask_size,
            )

        # loss
        if self._loss_function == 'dot':
            self.loss = DotLoss()
        elif self._loss_function == 'mse':
            self.loss = torch.nn.MSELoss()
        else:
            raise NotImplementedError(f'Unsupported loss function: {self._loss_function}')

        # metrics
        self.val_f1 = DetectionF1Metric(correct_distance=5.0)
        self.test_f1 = DetectionF1Metric(correct_distance=5.0)

        self.val_count = CountingMetric(self._output_channels)
        self.test_count = CountingMetric(self._output_channels)

        self.save_hyperparameters()

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, out_x2, out_x4 = self.network(x)

        out = nn.functional.interpolate(out, size=(self._mask_size[1], self._mask_size[0]), mode='bilinear', align_corners=True)
        out_x2 = nn.functional.interpolate(out_x2, size=(self._mask_size[1], self._mask_size[0]), mode='bilinear', align_corners=True)
        out_x4 = nn.functional.interpolate(out_x4, size=(self._mask_size[1], self._mask_size[0]), mode='bilinear', align_corners=True)
        
        return out, out_x2, out_x4

    def calculate_loss(self, y_pred, mask, gt, is_stage=False):
        if self._loss_function == 'dot':
            return self.loss(y_pred, self.postprocessing(y_pred, thresh=min(0.1 + (self.current_epoch/25), self._obj_threshold)), mask, gt, is_stage=is_stage)
        elif self._loss_function == 'mse':
            return self.loss(y_pred, mask), {}
        else:
            raise NotImplementedError(f'Unsupported loss function: {self._loss_function}')

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> Optional[torch.Tensor]:
        image, mask, gt, alt = batch
        
        out, out_x2, out_x4 = self.forward(image)
        
        if self._loss_function == 'mse':
            mask *= 500

        loss_x2, loss_d = self.calculate_loss(out, mask, gt)
        loss_x4, _ = self.calculate_loss(out_x2, mask, gt, is_stage=True)
        loss_x8, _ = self.calculate_loss(out_x4, mask, gt, is_stage=True)

        loss = loss_x2 + loss_x4 * 1/2 + loss_x8 * 1/4
         
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)

        for key, value in loss_d.items():
            self.log(f'train_{key}', value, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        image, mask, gt, alt = batch
            
        y_pred, _, _ = self.forward(image)
        
        assert y_pred.shape == mask.shape, f'Predicted shape: {y_pred.shape}, mask shape: {mask.shape}'
        
        predicted_points = self.postprocessing(y_pred)

        f1, precission, recall = self.val_f1(predicted_points, gt)
        self.log('val_f1', f1, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_P', precission, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_R', recall, on_step=False, on_epoch=True, sync_dist=True)

        mae, mae_norm = self.val_count(predicted_points, gt)
        self.log('val_mae', mae, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_mae_norm', mae_norm, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        image, mask, gt, alt, _, _ = batch

        y_pred, _, _ = self.forward(image)

        predicted_points = self.postprocessing(y_pred)

        f1, precission, recall = self.test_f1(predicted_points, gt)
        self.log('test_f1', f1, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_P', precission, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_R', recall, on_step=False, on_epoch=True, sync_dist=True)

        mae, mae_norm = self.test_count(predicted_points, gt)
        self.log('test_mae', mae, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_mae_norm', mae_norm, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self._lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self._train_steps//2,
        )

        schedule = {
            'scheduler': scheduler,
            'interval': 'step',
        }
        return [optimizer], [schedule]

    def postprocessing(self, y_pred_raw: torch.Tensor, thresh: float = None) -> torch.Tensor:
        if self._loss_function == 'mse':
            y_pred = y_pred_raw / 500.
        else:
            y_pred = torch.sigmoid(y_pred_raw)
            
        y_pred = self._nms(y_pred)

        return_values = []

        thresh = self._obj_threshold if thresh is None else thresh

        for batch_id in range(y_pred.shape[0]):
            pred_b = y_pred[batch_id]

            predictions = []

            for class_id in range(pred_b.shape[0]):
                yx = torch.argwhere(pred_b[class_id] > thresh)

                if yx.shape[0] > 0:
                    predictions.append(
                        torch.cat([
                            torch.full((yx.shape[0], 1), class_id, dtype=torch.int32).to(
                                y_pred.device),
                            yx[:, [1, 0]],
                            y_pred_raw[batch_id][class_id][yx[:, 0],
                                                           yx[:, 1]][:, None],
                        ], 1),
                    )

            if len(predictions) > 0:
                predictions = torch.cat(predictions, dim=0)
            else:
                predictions = torch.zeros(
                    (0, 4), dtype=torch.float32).to(y_pred.device)
                
            predictions[:, 1] *= self._image_size[0] / self._mask_size[0]
            predictions[:, 2] *= self._image_size[1] / self._mask_size[1]
                
            return_values.append(predictions)

        return return_values

    @staticmethod
    def _nms(heat, kernel=3):
        pad = (kernel - 1) // 2

        hmax = torch.nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()
        
        return heat * keep
