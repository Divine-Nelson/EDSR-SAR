# Deep Learning-Based Super-Resolution for SAR Images

## Overview
This project focuses on enhancing the spatial resolution of **Synthetic Aperture Radar (SAR)** images using deep learning techniques.

SAR images are widely used in remote sensing because they can operate under all weather conditions and during both day and night. However, high-resolution SAR data is often expensive and limited. This project addresses that problem by reconstructing **high-resolution (HR)** images from **low-resolution (LR)** inputs using AI-based super-resolution methods.

---

## Objectives
- Develop and evaluate AI-based super-resolution models for SAR images
- Compare deep learning approaches with traditional interpolation methods
- Improve reconstruction quality using advanced architectures
- Evaluate performance using quantitative metrics (PSNR, SSIM)

---

## Methods Implemented
The project follows an **iterative (spiral) development approach**:

###  Baseline
- Bicubic Interpolation

###  Deep Learning Models
- SRCNN (Super-Resolution CNN)
- Residual SRCNN
- EDSR (Enhanced Deep Super-Resolution Network)  *Final model*

---

## ⚙️ System Pipeline
1. Input HR SAR images  
2. Generate LR images via downsampling  
3. Extract random patches (256×256)  
4. Train deep learning models  
5. Reconstruct HR images  
6. Evaluate using PSNR and SSIM  

---

## Results
The EDSR model achieved the best performance:

| Method | PSNR (dB) | SSIM |
|------|--------|------|
| Bicubic | 27.65 ± 4.80 | 0.60 |
| EDSR (Proposed) | **28.15 ± 4.90** | **0.64** |

### Key Observations:
- Deep learning models outperform bicubic interpolation
- EDSR provides better edge reconstruction and structural preservation
- Performance is consistent across multiple SAR scenes

---

## Evaluation Metrics
- **PSNR (Peak Signal-to-Noise Ratio)** → measures reconstruction accuracy  
- **SSIM (Structural Similarity Index)** → measures perceptual similarity  

---
