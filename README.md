## Why DOT Detection?

The DOT approach defines objects as pairs of coordinates and uses an encoder-decoder architecture to generate precise object masks and coordinates. The proposed Pixel Distill module, designed to distill essential information from high-resolution images, further enhances the processing of high-definition images, leading to improved object detection accuracy.

## Results

The repository is a supplement to the paper "More Pixels, More Precision: Enhancing People Localisation in Drone Imagery Using Dot Approach and Pixel Distill Module". The method achieves state-of-the-art results on the DroneCrowd and newly introduced UP-COUNT datasets. The results are summarized in the table below.

| **Model** | Dronecrowd |         | UP-COUNT |         |
|:---------:|:----------:|:-------:|:--------:|:-------:|
|           |    L-mAP   | L-AP@10 |   L-mAP  | L-AP@10 |
|   [STNNet](https://github.com/VisDrone/DroneCrowd/tree/master/STNNet)  |   40.45    |   42.75  |  37.20  |  28.48       |
|    [MFA](https://github.com/asanomitakanori/mfa-feature-warping)    |   43.43    |   47.14   |   x   |    x    |
|  [STEERER](https://github.com/taohan10200/STEERER)  |    38.31    |   41.96  |  40.20   | 42.14   |
|    [RFLA](https://github.com/Chasel-Tsui/mmdet-rfla)   |   32.05     |  34.41   |   32.41  |  33.27  |
|  [SD-DETR](https://github.com/kai271828/SD-DERT)  |    48.12    |  52.56   |  57.89    |  63.57    |
|    DOT    |   47.63    |  53.37   |   60.66  |  69.07   |
|  DOT + PD |   **51.00**      |  **57.06**   | **66.49**   |  **75.46**  |


## Dataset

* The DroneCrowd dataset can be downloaded from [here](https://github.com/VisDrone/DroneCrowd/tree/master#dronecrowd-full-version)

* The UP-COUNT dataset is available [here](https://up-count.github.io/)


## Usage

### Clone the repository (including submodules)

```
git clone --recurse-submodules https://github.com/up-count/uav-dot.git
```

### Requirements

Install requirements with: 

```bash
pip install -r requirements.txt
```

```bash
pip install -r ./eval_tool/requirements.txt
```


### Checkpoints

Download the checkpoints from the links below and place them in the `./checkpoints` directory.

| **Model** | **Dataset** | **L-mAP** | **L-AP@10** |   **Link**   |
|:---------:|:-----------:|:---------:|:-----------:|:------------:|
|    DOT    |  Dronecrowd |   47.63   |   53.37     | [download](https://drive.google.com/file/d/1jHZ2_85kS4tdG5Qbq3Jn0Xpjs8So6mwK/view?usp=sharing) |
|  DOT + PD |  Dronecrowd |   51.00   |   57.06     | [download](https://drive.google.com/file/d/1wYa01jGYfrAun3SfxuWzcKAni3hzBOMV/view?usp=sharing) |
|    DOT    |   UP-COUNT  |   60.66   |   69.07     | [download](https://drive.google.com/file/d/16MghcySpCxS0OxJzTJyRLr3AZKZ7cP0w/view?usp=sharing) |
|  DOT + PD |   UP-COUNT  |   66.49   |   75.46     | [download](https://drive.google.com/file/d/1K-SkfIPbivnOw7atjRQHW11Bt0-bhcKi/view?usp=sharing) |

### Training

```bash
python3 main.py --config-name dronecrowd.yaml
```

```bash
python3 main.py --config-name upcount.yaml
```

### Testing and evaluation

* Generate predictions on the test set:

> **Note** If you want to use DOT without Pixel Distill, you need to add the `spatial_mode=interpolate` flag before the `--config-name` argument and update the `restore_from_ckpt` argument.

```bash
python3 pt_pred.py restore_from_ckpt=./checkpoints/dot_pd_dronecrowd_51.00.ckpt --config-name dronecrowd.yaml
```

```bash
python3 pt_pred.py restore_from_ckpt=./checkpoints/dot_pd_upcount_66.49.ckpt --config-name upcount.yaml
```


* Evaluate the predictions:

```bash
python3 eval_tool/eval.py -d dronecrowd -p ./results/pt_pred/dronecrowd/
```

```bash
python3 eval_tool/eval.py -d upcount -p ./results/pt_pred/upcount/
```

## Infer on your own video

```bash
python3 infer_video.py restore_from_ckpt=./checkpoints/dot_pd_upcount_66.49.ckpt +video=<PATH_TO_VIDEO> --config-name upcount.yaml
```

Results will be saved in the `./infer_results/` directory.


## Citation

If you find this repository useful, please consider citing the following paper:
```
Soon ;)
```
