import cv2
import hydra
import numpy as np
from tqdm import tqdm

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

    if cfg.restore_from_ckpt is None:
        raise ValueError('Please provide a checkpoint to restore from.')
        
    model = DotRegressor.load_from_checkpoint(
        checkpoint_path=cfg.restore_from_ckpt
    )
    
    Path('./infer_results').mkdir(parents=True, exist_ok=True)
    
    if 'video' not in cfg:
        raise ValueError('Please provide a video path to infer from using video=<path>.')
    
    viz = 'viz' in cfg
        
    if viz:
        cv2.namedWindow('image')
    
    model.eval().cuda()
    
    cap = cv2.VideoCapture(cfg.video)
    
    new_name = cfg.video.split("/")[-1].split(".")[0] + '.avi'
    org_shape = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                 
    writter = cv2.VideoWriter(
        f'./infer_results/pred_{new_name}',
        cv2.VideoWriter_fourcc(*'XVID'),
        cap.get(cv2.CAP_PROP_FPS),
        (org_shape[0], org_shape[1])
    )
    
    if not cap.isOpened():
        raise ValueError('Video not found.')
    
    for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = cap.read()
        
        if not ret:
            print('Video ended.')
            break

        
        # preprocess
        image = frame.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (cfg.image_size[0], cfg.image_size[1]))
        image = image * 1. / 255.
        image = (image - cfg.data_mean) / cfg.data_std
        
        with torch.no_grad():
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
                
            y_pred, _, _ = model(image.cuda())
                    
            predicted_points = model.postprocessing(y_pred)[0]
        
        xs = predicted_points[:, 1].cpu().numpy()
        ys = predicted_points[:, 2].cpu().numpy()
        
        xs = xs * org_shape[0] / cfg.image_size[0]
        ys = ys * org_shape[1] / cfg.image_size[1]
        
        for x, y in zip(xs, ys):
            cv2.circle(frame, (int(x), int(y)), 7, (0, 0, 255), 3)
        
        writter.write(frame)
        
        if viz:
            cv2.imshow('image', cv2.resize(frame, (1280, 720)))
            key = cv2.waitKey(1)
            
            if key == ord('q'):
                return
            
    cap.release()
    writter.release()
    print(f'Saved video to ./infer_results/pred_{new_name}')
    
    if viz:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
