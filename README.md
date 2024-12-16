# Image Super-Resolution using GAN & Diffusion
Zhaowen Gu, Sergio Rodriguez, Nina Servan-Schreiber
{petergu, sergioer, nsrvs}@bu.edu

# Presentation slides: 
https://docs.google.com/presentation/d/1S8Cxx1bEnq71VE9vJzZUV9hs2sQjT7Miae5eAxaPQeY/edit#slide=id.p

##### Not all models and files utilized are currently in this repository

### Task
This project introduces a two-stage single-image super-resolution (SISR) framework using a CycleGAN for high-to-low degradation and a ResShift diffusion model for low-to-high super-resolution. The CycleGAN approach in the first stage enables realistic simulation of real-world image degradation (e.g., blur, noise) in an unpaired setting, ensuring that degraded images retain content integrity through cycle-consistency. In the second stage, the low-to-high diffusion model reconstructs high resolution (HR) images from the low resolution (LR) ones. Our goal is to assess whether the performance of the diffusion model can be improved by training on realistically degraded LR images.

### Related Work
Early GAN-based methods like SRGAN [1] showed promise for SISR by achieving realistic details but often relied on synthetic down-sampling, limiting their effectiveness on real-world images. Bulat et al. [2] addressed this gap with a two-stage GAN approach, introducing a high-to-low degradation network to model realistic degradation. Building on these efforts, we integrate CycleGAN [5] for unpaired degradation modeling with a diffusion model for the super-resolution stage. Diffusion models, such as SRDiff [4] and ResShift [3], have recently gained attention for their stability and ability to recover fine details, even in challenging conditions like low-light and motion blur. This combination of CycleGan for degradation and ResShift diffusion for restoration leverages the strengths of both methods to enhance real-world SISR performance.

### Approach
High-to-Low GAN Network: We will use the CycleGAN network trained on unpaired real-world LR and HR images to degrade HR images in a realistic way.
Low-to-High Diffusion Network: We will use the ResShift diffusion model trained on paired real-world HR and generated LR images to address the SISR task. 
Baseline: We will compare our two-stage SISR framework with a single-stage SISR framework, i.e., using only the diffusion network trained on HR real-world images and their “naive” LR equivalent generated through simple bilinear downsampling.

### Dataset ans Metrics
Dataset: We use two real-world face datasets: CelebA for HR and Tinyface for LR. These two real-world datasets (unpaired) will be used to train the high-to-low CycleGAN network. We will then generate an artificial LR dataset from the real-world HR dataset using our trained CycleGAN network. These two datasets (paired) will be used to train the low-to-high ResShift diffusion model. We will test the performance of our diffusion model on both real-world and artificial LR images.  
Metrics: We will use perceptual quality metrics such as Multi-Scale Image Quality Transformer (MUSIQ), Peak Signal-to-Noise Ratio (PSNR), Mean Opinion Score (MOS), and Structural Similarity Index Measure (SSIM).

### References
1. C. Ledig, L. Theis, F. Huszár, J. Caballero, A. Cunningham, A. Acosta, A. Aitken, A. Tejani, J. Totz, Z. Wang, W. Shi. Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network. CVPR, 2017.

1. A. Bulat, J. Yang, G. Tzimiropoulos. To Learn Image Super-Resolution, Use a GAN to Learn How to Do Image Degradation First. ECCV, 2018.

1. Zongsheng Yue, Jianyi Wang, and Chen Change Loy. ResShift: Efficient Diffusion Model for Image Super-Resolution by Residual Shifting. Proceedings of the 37th Conference on Neural Information Processing Systems (NeurIPS), 2023

1. Haoying Li, Yifan Yang, Meng Chang, Shiqi Chen, Huajun Feng, Zhihai Xu, Qi Li, and Yueting Chen. SRDiff: Single Image Super-Resolution with Diffusion Probabilistic Models. Neurocomputing, vol. 479, pp. 47–59, 2022

1. Zhu, Jun-Yan, et al. "Unpaired image-to-image translation using cycle-consistent adversarial networks." Proceedings of the IEEE international conference on computer vision. 2017.
