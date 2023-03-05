# Backdoor-Defense-Via-Suppressing-Model-Shortcuts

This is the official implementation of our paper [Backdoor Defense Via Suppressing Model Shortcuts](https://www.researchgate.net/publication/365299231_Backdoor_Defense_via_Suppressing_Model_Shortcuts), accepted by the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2023. This research project is developed based on Python 3 and Pytorch, created by [Sheng Yang](https://scholar.google.com/citations?view_op=list_works&hl=zh-CN&hl=zh-CN&user=HZdisxYAAAAJ) and [Yiming Li](http://liyiming.tech/).

## Reference

If our work or this repo is useful for your research, please cite our paper as follows:

```
@inproceedings{yang2023backdoor,
  title={Backdoor Defense via Suppressing Model Shortcuts},
  author={Yang, Sheng and Li, Yiming and Jiang, Yong and Xia, Shu-Tao},
  booktitle={ICASSP},
  year={2023}
}
```

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Dataset Preparation

Make sure the directory `data` follows:

```File
data
├── cifar10  
│   ├── train
│   └── test
├── gtsrb
|   ├── train
│   └── test
```

> 📋 Data Download Link:  
> [data](https://drive.google.com/drive/folders/1tPppCn2VQ89Jy_LT-1CMbV2xPgV6O5p1?usp=sharing)

## Train Backdoor Model

```
python train_backdoormodel.py --model=resnet18 --datasets=cifar10 --attack=badnets --target=1 --poisoned_rate=0.05 --mp=experiments --gpu=0
```

## Shortcut Selection

```
python chooselayer.py --model=resnet18 --datasets=cifar10 --model_path=filepath --pdp=filepath --gpu=0
```

## Shortcut Suppression With Finetuning

```
python SSFT.py --model=resnet18 --datasets=cifar10 --model_path=filepath --pdp=filepath --target=1 --layer=layer3 --location=layer3.1.shortcyt --gpu=0 
```
