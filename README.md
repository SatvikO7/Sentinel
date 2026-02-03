# Image Generation & Super-Resolution Models

This repository contains a comparative implementation and analysis of three deep learning models for image processing tasks:

Diffusion Model (Transformer + GAN based)

Generative Adversarial Network (GAN)

CNN / ResNet-based Model

The objective is to evaluate these models based on performance, training efficiency, and output quality.
Among all models, the GAN model demonstrated the best overall performance.

Models Included
1. Diffusion Model

Combines Diffusion (DDPM), Transformer (SwinIR), and GAN techniques

Produces high-quality outputs

Computationally expensive and slow to train

2. GAN Model (Best Performing)

Standard Generator–Discriminator architecture

Adversarial training for realistic image generation 

Faster convergence and efficient training

Why GAN performed best:

Sharp and realistic outputs

Faster training compared to diffusion

Lower computational cost

Simple and effective architecture


3. CNN / ResNet Model

Uses convolutional layers with residual connections

Suitable for feature learning and classification

Not designed for image generation.

![GAN image](utils/LR_HR_SR_comparasion.png)
GAN Comparision image

![ResNet comparision image](utils/comparision.png)
ResNet comparision image

![Difussion model image](utils/swinir_visual_results.png)
Diffusion model image

![ResNet Graphs](utils/graphs.png)
Resnet Performance Graph

![Diffusion Graphs](utils/swinir_training_history.png)
Diffusion Performance Graph

![Samples](utils/samples.png)
Data Samples 



| Model     | Image Generation | Training Speed | Output Quality | Overall    |
| --------- | ---------------- | -------------- | -------------- | ---------- |
| Diffusion | Yes              | Slow           | High           | Medium     |
| **GAN**   | **Yes**          | **Fast**       | **Very High**  | **Best** |
| ResNet    | No               | Fast           | Moderate       | Low        |




# Tech Stack

PyTorch / TensorFlow

torch

torchvision

diffusers

accelerate

transformers

earthengine-api

rasterio

opencv-python

numpy

matplotlib

tqdm

# Dataset

1. Metadata CSV file

There is a metadata table (usually a .csv file). This is a spreadsheet-like file that tells you important info about every image in the dataset:

Location name / ID of each image pair

Latitude & longitude coordinates

Which images are high resolution and which are low resolution

Possible labels like land type (urban, forest, water, etc.)

Date and time the images were taken

This is the index of all the data — you use it to know where an image comes from, what resolution it has, and how everything is matched.

2. High-resolution (HR) image folder

These files are the sharp, detailed satellite images in the dataset.

Usually in a folder named something like HR, high_res, or similar

They are 1.5 meters per pixel — meaning they show the Earth at about 1.5 m detail

These images are often from Airbus satellites

They may be stored as GeoTIFF (.tif) files (a common format for satellite images)

Each file is one big patch of land at high detail used as the ground truth

Example: location123_highres.tif

3. Low-resolution (LR) image folder

These are the lower-detail satellite images that are paired with the high-resolution ones.

Often in a folder named LR, low_res, or similar

These come from Sentinel-2 satellites (free public data)

They are 10 meters per pixel — much blurrier than HR

For each high-resolution image, there are usually several LR images taken at different times (called temporal revisits)

These allow multi-frame super-resolution training.

4. Matching pairs structure

The dataset is organized so that each high-resolution image has matching low-resolution images:

A matching group of 1 HR + 8 LR files

The metadata CSV tells you which belong together

This lets you train models that take multiple LR images and predict the HR image


# How to Run locally

Clone the repository

Open any notebook in Jupyter or Google Colab

Install required dependencies

Run cells sequentially

# Conclusion

While diffusion models provide strong theoretical results, their high computational cost limits practical use.
The GAN model achieved the best balance of quality, speed, and efficiency, making it the most effective model in this project.
