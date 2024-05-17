import cv2
import hydra
import numpy as np

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from pathlib import Path

from src.model.dot_regressor import DotRegressor
from src.datamodule.dot_datamodule import DotDatamodule



@hydra.main(version_base=None, config_path='./configs/')
def main(cfg: DictConfig) -> None:
    pl.seed_everything(seed=42)

    datamodule = DotDatamodule(
        root_data_path=cfg.data_path,
        dataset=cfg.dataset,
        data_fold=cfg.data_fold,
        batch_size=1,
        workers=cfg.workers,
        data_mean=cfg.data_mean,
        data_std=cfg.data_std,
        image_size=cfg.image_size,
        mask_size=cfg.mask_size,
        mosaic=cfg.mosaic,
    )

    if cfg.restore_from_ckpt is None:
        raise ValueError('Please provide a checkpoint to restore from.')
        
    model = DotRegressor.load_from_checkpoint(
        checkpoint_path=cfg.restore_from_ckpt
    )
    
    viz = 'viz' in cfg
        
    if viz:
        cv2.namedWindow('image')
    
    model.eval().cuda()
    
    list_path = []
    
    if cfg.dataset == 'dronecrowd':
        org_shape = (1920, 1080)
    elif cfg.dataset == 'upcount':
        org_shape = (3840, 2160)
    
    Path(f'./results/pt_pred/{cfg.dataset}').mkdir(parents=True, exist_ok=True)
    
    for i, batch in enumerate(tqdm(datamodule.test_dataloader())):
        with torch.no_grad():
            image, masks, annos, alt, shapes, pathes = batch
            
            if viz:
                if cfg.dataset == 'dronecrowd':
                    if i%100 != 0:
                        continue
            
            y_pred, _, _ = model(image.cuda())
                
            predicted_points = model.postprocessing(y_pred)[0]
                                    
            # # scale to mask size            
            predicted_points[:, 3] = torch.sigmoid(predicted_points[:, 3])
            
            xs = predicted_points[:, 1].cpu().numpy()
            ys = predicted_points[:, 2].cpu().numpy()
            confs = predicted_points[:, 3].cpu().numpy()
            
            xs = xs * org_shape[0] / cfg.image_size[0]
            ys = ys * org_shape[1] / cfg.image_size[1]
            
            pts = np.hstack(
                (
                    np.expand_dims(xs.T, axis=1),
                    np.expand_dims(ys.T, axis=1),
                    np.expand_dims(confs.T, axis=1),
                )
            )  
            
            if viz:                
                image = image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
                image = np.uint8((image*cfg.data_std+cfg.data_mean) * 255)
                image = np.ascontiguousarray(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image = cv2.resize(image, org_shape)
            
                for pt in pts:
                    cv2.circle(image, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)    
            
            np.savetxt(f'./results/pt_pred/{cfg.dataset}/' + pathes[0].replace('.jpg', '_loc.txt'), pts)
            
            seq = pathes[0].split('/')[-1].split('__')[0]
            list_path.append(seq[3:6] if cfg.dataset == 'dronecrowd' else seq)
            
            if viz:
                cv2.imshow('image', cv2.resize(image, (1280, 720)))
                key = cv2.waitKey(1)
                
                if key == ord('q'):
                    return
            
    list_path = np.unique(list_path)
    
    with open(f'./results/pt_pred/{cfg.dataset}_test_sequences.txt', 'w') as f:
        for item in list_path:
            f.write("%s\n" % item)

if __name__ == '__main__':
    main()
