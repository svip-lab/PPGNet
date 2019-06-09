# PPGNet: Learning Point-Pair Graph for Line Segment Detection

PyTorch implementation of our CVPR 2019 paper:

[**PPGNet: Learning Point-Pair Graph for Line Segment Detection**](https://arxiv.org/pdf/1905.03415)

Ziheng Zhang*, Zhengxin Li*, Ning Bi, Jia Zheng, Jinlei Wang, Kun Huang, Weixin Luo, Yanyu Xu, Shenghua Gao

(\* Equal Contribution)

![arch](https://www.researchgate.net/profile/Ziheng_Zhang3/publication/332977700/figure/fig2/AS:756853751427073@1557459404667/The-PPGNet-architecture-First-the-backbone-computes-shared-features-of-size-C-H-4-W_W640.jpg)

## Usage

1. download the preprocessed *SIST-Wireframe* dataset here (still being uploaded).
2. specify dataset path in the `train.sh` script.
3. run `train.sh`

Please note that the code requires the GPU memory to be at least 24GB. For GPU with memory smaller than 24GB, you can change the `----block-inference-size` parameter in `train.sh` to be a smaller integer to avoid the out-of-memory error.

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
