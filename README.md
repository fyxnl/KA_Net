# Advancing Real-World Image Dehazing: Perspective, Modules, and Training - TPAMI 2024

This is the official PyTorch implementation of KA_Net.  
## Abstract:
Restoring high-quality images from degraded hazy observations is a fundamental and essential task in the field of computer vision. While deep models have achieved significant success with synthetic data, their effectiveness in real-world scenarios remains uncertain. To improve adaptability in real-world environments, we construct an entirely new computational framework by making efforts from three key aspects: imaging perspective, structural modules, and training strategies. To simulate the often-overlooked multiple degradation attributes found in real-world hazy images, we develop a new hazy imaging model that encapsulates multiple degraded factors, assisting in bridging the domain gap between synthetic and real-world image spaces. In contrast to existing approaches that primarily address the inverse imaging process, we design a new dehazing network following the “localization-and-removal” pipeline. The degradation localization module aims to assist in network capture discriminative haze-related feature information, and the degradation removal module focuses on eliminating dependencies between features by learning a weighting matrix of training samples, thereby avoiding spurious correlations of extracted features in existing deep methods. We also define a new Gaussian perceptual contrastive loss to further constrain the network to update in the direction of the natural dehazing. Regarding multiple full/no-reference image quality indicators and subjective visual effects on challenging RTTS, URHI, and Fattal real hazy datasets, the proposed method has superior performance and is better than the current state-of-the-art methods.


See more details in [[paper]](https://ieeexplore.ieee.org/document/10564179)

## Environment:

- Windows: 10

- CUDA Version: 11.0 
- Python 3.7

## Dependencies:

- torch==1.7.0
- torchvision==0.7.0
- NVIDIA GPU and CUDA

## Pretrained Weights & Dataset

1. Download [Dehaze weights](https://pan.baidu.com/s/1HETnxLCTxjHRsBg2STwsHQ) and Extraction code: [n2k4]
2. We release part of the data used for training, please download them if you need.( https://pan.baidu.com/s/1kaKo06PGAHjEQIMfNAgbkQ?pwd=sw6s) and Extraction code:[sw6s]
3. (**Note that the final complete version of the data and synthetic code will be released soon.) 

## Test

Our test run is simple, just change the input and output paths according to your requirements

```
python KA_net_test.py
```
## Train

Please note that due to time, our training code is not fully sorted out yet, but it won't take long, so stay tuned


## Qualitative Results

Look at the output in the output folder


## Acknowledgement

We thank the authors of [Transweather](https://arxiv.org/abs/2111.14813). Part of our code is built upon their modules.

 
## Citation

If our work helps your research, please consider to cite our paper:

Y. Feng, L. Ma, X. Meng, F. Zhou, R. Liu and Z. Su, "Advancing Real-World Image Dehazing: Perspective, Modules, and Training," in IEEE Transactions on Pattern Analysis and Machine Intelligence, doi: 10.1109/TPAMI.2024.3416731

 


