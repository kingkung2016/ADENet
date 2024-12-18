# ADENet
Adaptive Depth Enhancement Network for RGB-D Salient Object Detection

This is the official implementation of "Adaptive Depth Enhancement Network for RGB-D Salient Object Detection" as well as the follow-ups. The paper has been published by IEEE Signal Processing Letters, 2025. The paper link is [here](https://ieeexplore.ieee.org/document/10767761/).
****

## Content
* [Run ADENet code](#Run-ADENet-code)
* [Pretrained models](#Pretrained-models)
* [Saliency maps](#Saliency-maps)
* [Evaluation tools](#Evaluation-tools)
* [Citation](#Citation)
****

## Run ADENet code
- Train <br>
  run `python train.py` <br>
  \# put pretrained models in the pretrained folder <br>
  \# set '--train-root' to your training dataset folder
  
- Test <br>
  run `python test.py` <br>
  \# set '--test-root' to your test dataset folder <br>
  \# set '--ckpt' as the correct checkpoint <br>
****

## Pretrained models
  - The pretrained models can be downloaded in [Baidu Cloud](https://pan.baidu.com/s/1u5dv6crd7e4OwM9hBdooWg) (fetach code is pcmi). Then put the pretrained models such as 'resnet_50.pth' in the pretrained folder.
****

## Saliency maps
  - The saliency maps can be approached in [Baidu Cloud](https://pan.baidu.com/s/1Rz0iiwmA6QsCnPDqzPFnrg) (fetach code is ader).
  - The saliency maps can be approached in [Baidu Cloud](https://pan.baidu.com/s/1SNeglGTt-qZeuv_z5wtSsA) (fetach code is adet).
  - Note that all testing results are provided not only including those listed in the paper.
****

## Evaluation tools
- The evaluation tools, training and test datasets can be obtained in [RGBD-SOD-tools](https://github.com/kingkung2016/RGBD-SOD-tools).
****

## Citation
```
@ARTICLE{yi2025adaptive,
  author={Yi, Kang and Li, Yumeng and Tang, Haoran and Xu, Jing},
  journal={IEEE Signal Processing Letters}, 
  title={Adaptive Depth Enhancement Network for RGB-D Salient Object Detection}, 
  year={2025},
  volume={32},
  number={},
  pages={176-180},
  doi={10.1109/LSP.2024.3506863}}
}

```
****



