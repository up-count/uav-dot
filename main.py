import os

import hydra

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import NeptuneLogger
from torch.distributed.algorithms.ddp_comm_hooks import (
    default_hooks as default,
)

from src.model.dot_regressor import DotRegressor
from src.datamodule.dot_datamodule import DotDatamodule


@hydra.main(version_base=None, config_path='./configs/')
def main(cfg: DictConfig) -> None:
    pl.seed_everything(seed=42)

    datamodule = DotDatamodule(
        root_data_path=cfg.data_path,
        dataset=cfg.dataset,
        data_fold=cfg.data_fold,
        batch_size=cfg.batch_size,
        workers=cfg.workers,
        data_mean=cfg.data_mean,
        data_std=cfg.data_std,
        image_size=cfg.image_size,
        mask_size=cfg.mask_size,
        mosaic=cfg.mosaic,
    )

    if cfg.restore_from_ckpt is not None:
        print('Restoring entire state from checkpoint...')
        model = DotRegressor.load_from_checkpoint(
            checkpoint_path=cfg.restore_from_ckpt
        )
    else:
        print('Creating new model...')
        model = DotRegressor(
            encoder_name=cfg.encoder_name,
            input_channels=cfg.input_channels,
            output_channels=cfg.output_channels,
            spatial_mode=cfg.spatial_mode,
            loss_function=cfg.loss,
            lr=cfg.lr,
            train_steps=datamodule.get_train_steps(),
            visualize_test_images=cfg.visualize_test_images if not cfg.debug else False,
            obj_threshold=cfg.obj_threshold,
            image_size=cfg.image_size,
            mask_size=cfg.mask_size,
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir'], 
        filename='epoch_{epoch}-f1_{val_f1:.2f}', 
        monitor=cfg.monitor, 
        auto_insert_metric_name=False, 
        verbose=True, 
        mode=cfg.monitor_mode)
    
    model_summary_callback = ModelSummary(max_depth=1)
    
    early_stopping_callback = EarlyStopping(
        monitor=cfg.monitor,
        mode=cfg.monitor_mode,
        patience=cfg.es_patience)
    
    lr_monitor = LearningRateMonitor(logging_interval='step')

    if not cfg.debug:
        logger = NeptuneLogger(
            api_key=os.environ['NEPTUNE_API_TOKEN'],
            project='Vision/UP-COUNT',
            log_model_checkpoints=True,
        )
    else:
        logger = None

    callbacks = [
        checkpoint_callback,
        model_summary_callback,
        early_stopping_callback,
    ]

    if not cfg.debug:
        callbacks.append(lr_monitor)

    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        devices="auto" if cfg.devices <= 0 else cfg.devices,
        accelerator='gpu' if torch.cuda.is_available() and cfg.devices > 0 else 'cpu',
        precision=cfg.precision,
        max_epochs=cfg.epochs,
        benchmark=True if cfg.devices > 0 else False,
        sync_batchnorm=True if torch.cuda.is_available() else False,
    )

    if not cfg.test_only:
        trainer.fit(model, datamodule)
        trainer.test(model, datamodule, ckpt_path='best')
    else:
        assert cfg.restore_from_ckpt is not None
        trainer.test(model, datamodule)


if __name__ == '__main__':
    main()
