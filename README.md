# Conditional Diffusion Model (DDPM + DDIM)

## Overview
This project implements a conditional diffusion model from scratch for image generation on the CIFAR-10 dataset. It covers both the forward noise addition process and the reverse denoising process using a UNet-based architecture.

## Features
- Conditional UNet with self-attention and time embeddings  
- Forward diffusion with noise scheduling  
- Reverse denoising using DDPM  
- Accelerated sampling using DDIM  
- Classifier-Free Guidance for conditional generation  

## Tech Stack
- Python  
- PyTorch  
- NumPy, Matplotlib  
- CIFAR-10 dataset  

## Learning Outcomes
- Understanding diffusion processes and noise scheduling  
- Implementing UNet architectures for denoising tasks  
- Exploring differences between DDPM and DDIM sampling  
- Studying the effect of conditional guidance in generation  

## Running the Project
Install dependencies and run the training script:

```bash
pip install torch torchvision matplotlib tqdm
python your_script.py
