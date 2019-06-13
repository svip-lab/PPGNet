# PPGNet: Learning Point-Pair Graph for Line Segment Detection

PyTorch implementation of our CVPR 2019 paper:

[**PPGNet: Learning Point-Pair Graph for Line Segment Detection**](https://www.aiyoggle.me/publication/ppgnet-cvpr19/ppgnet-cvpr19.pdf)

Ziheng Zhang*, Zhengxin Li*, Ning Bi, Jia Zheng, Jinlei Wang, Kun Huang, Weixin Luo, Yanyu Xu, Shenghua Gao

(\* Equal Contribution)

\* New: The poster can be found [HERE](https://www.aiyoggle.me/publication/ppgnet-cvpr19).


![arch](https://www.researchgate.net/profile/Ziheng_Zhang3/publication/332977700/figure/fig2/AS:756853751427073@1557459404667/The-PPGNet-architecture-First-the-backbone-computes-shared-features-of-size-C-H-4-W_W640.jpg)

## Requirements
- Python >= 3.6
- fire >= 0.1.3
- numba >= 0.40.0
- numpy >= 0.15.0
- pytorch >= 0.4.1
- scikit-learn >= 0.19.1
- scipy >= 1.1.0
- tensorboard >= 1.11.0
- tensorboardX >= 1.4
- torchvision >= 0.2.1

## Usage

1. clone this repository (and make sure you fetch all .pth files right with [git-lfs](https://git-lfs.github.com/)): `git clone https://github.com/svip-lab/PPGNet.git`
2. download the preprocessed *SIST-Wireframe* dataset from [BaiduPan](https://pan.baidu.com/s/1Sbdi1lL492fhmPL1t1Ov0w) (code:lnfp) or [Google Drive](https://drive.google.com/file/d/1KggPcHCRu8BcOqCvVZCXiB64y9L2nQDf/view?usp=sharing).
3. specify the dataset path in the `train.sh` script.
4. run `train.sh`.

Please note that the code requires the GPU memory to be at least 24GB. For GPU with memory smaller than 24GB, you can use a smaller batch with `--batch-size` parameter and/or change the `----block-inference-size` parameter in `train.sh` to be a smaller integer to avoid the out-of-memory error.

## Citation

Please cite our paper for any purpose of usage.
```
@inproceedings{zhang2019ppgnet,
  title={PPGNet: Learning Point-Pair Graph for Line Segment Detection},
  author={Ziheng Zhang and Zhengxin Li and Ning Bi and Jia Zheng and Jinlei Wang and Kun Huang and Weixin Luo and Yanyu Xu and Shenghua Gao},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```

