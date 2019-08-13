# AutoGAN: Neural Architecture Search for Generative Adversarial Networks
The implementation of [AutoGAN: Neural Architecture Search for Generative Adversarial Networks](https://arxiv.org/abs/1908.03835). 


## Introduction
We've desinged a novel neural architecture search framework for generative adversarial networks (GANs), dubbed AutoGAN. Experiments validate the effectiveness of AutoGAN on the task of unconditional image generation. Specifically, our discovered architectures achieve highly competitive performance on unconditional image generation task of CIFAR-10, which obtains a record FID score of **12.42**, a competitive Inception score of **8.55**. 

**RNN controller:**
<p align="center">
  <img src="imgs/ctrl.png" alt="ctrl" width="90%">
</p>

**Search space:**
<p align="center">
  <img src="imgs/ss.png" alt="ss" width="30%">
</p>

**Discovered network architecture:**
<p align="center">
  <img src="imgs/cifar_arch1.png" alt="cifar_arch1" width="75%">
</p>

## Performance
Unconditional image generation on CIFAR-10.
<p align="center">
  <img src="imgs/cifar10_res.png" alt="cifar10_res" width="45%">
</p>

Unconditional image generation on STL-10.
<p align="center">
  <img src="imgs/stl10_res.png" alt="stl10_res" width="45%">
</p>

## Set-up

### install libraries:
python >= 3.6
```bash
pip install -r requirements.txt
```

### prepare fid statistic file
 ```bash
mkdir fid_stat
 ```
Download the pre-calculated statistics
([Google Drive](https://drive.google.com/drive/folders/1UUQVT2Zj-kW1c2FJOFIdGdlDHA3gFJJd?usp=sharing)) to `./fid_stat`.

## How to train & test

### train
```bash
sh exps/autogan_cifar10_a.sh
```

### test
Run the following script:
```bash
python test.py \
--dataset cifar10 \
--img_size 32 \
--bottom_width 4 \
--model autogan_cifar10_a \
--latent_dim 128 \
--gf_dim 256 \
--g_spectral_norm False \
--load_path /path/to/*.pth \
--exp_name test_autogan_cifar10_a
```
Pre-trained models are provided ([Google Drive](https://drive.google.com/drive/folders/1IYDNrKY3m97K3bx_uIzOL6vFmCjGNpYZ?usp=sharing)).

## Citation
If you find this work is useful to your research, please cite our paper:
```bibtex
@InProceedings{Gong_2019_ICCV,
author = {Gong, Xinyu and Chang, Shiyu and Jiang, Yifan and Wang, Zhangyang},
title = {AutoGAN: Neural Architecture Search for Generative Adversarial Networks},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {Oct},
year = {2019}
}
```

## Acknowledgement
1. Inception Score code from [OpenAI's Improved GAN](https://github.com/openai/improved-gan/tree/master/inception_score) (official).
2. FID code and CIFAR-10 statistics file from [https://github.com/bioinf-jku/TTUR](https://github.com/bioinf-jku/TTUR) (official).

