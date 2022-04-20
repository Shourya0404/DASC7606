###### tags: `DASC7606 Deep learning Assignment`

# Assignment 2: Image-to-Image Translation
## 1 Introduction
In this assignment, you will implement generative adversarial networks (GANs), and apply them to image-to-image translation tasks. This system will translate one image to another correlative image.

### 1.1 What's image-to-image translation?

The Image-to-image translation is the task of taking images from one domain and translating them to another domain, so that they have the style (or characteristics) of images from another domain. For example, in the following pictures, image-to-image translation tasks are responsible for translating the semantic  images to the street scene images, or translating  aerial images to map images.
![](https://github.com/ChongjianGE/COMP3340_Applied_DL_Note/blob/main/demo.png?raw=true)

### 1.2 What will you learn from this assignment?
This assignment will walk you through the specific aerial-to-maps tasks. You can refer to the following picture for intuitive illustration. 
![](https://github.com/ChongjianGE/COMP3340_Applied_DL_Note/blob/main/2.jpg?raw=true)
Besides, you will train a GAN model from scratch, and you will also be asked to utilize the standard SSIM metrics for evaluation. 

The goals of this assignment are as follows:
- Understand the architecture of generative adversarial networks (GANs) and how they work to generate a realistic photo.
- Understand how the Generative model and the Discriminative model competes with each other in GAN.
- Implement a GAN model, and train them with the maps-to-aerial dataset.
- Understand and utilize the SSIM metrics for evaluation.

## 2 Setup
You can work on the assignment in one of two ways: locally on your own machine, or on a virtual machine on HKU GPU Farm.

### 2.1 Working remotely on HKU GPU Farm (Recommended)
**Note:** after following these instructions, make sure you go to **Working on the assignment** below (i.e., you can skip the Working locally section).

As part of this course, you can use HKU GPU Farm for your assignments. We recommend you follow the quickstart provided by the [**official website**](https://www.cs.hku.hk/gpu-farm/quickstart) to get familiar with HKU GPU Farm.

After checking the quickstart document, make sure you have gained the following skills:

- Knowing how to access the GPU Farm and use GPUs in interactive mode. We recommend using GPU support for this assignment, since your training will go much, much faster.
- Geting familar with running Jupyter Lab without starting a web browser.
- Knowing how to use tmux for unstable networks connections.

### 2.2 Working locally

If you have the GPU resources on your own PC/laptop. Here’s how you install the necessary dependencies:

**Installing GPU drivers (Recommend if work locally) :** If you choose to work locally, you are at no disadvantage for the first parts of the assignment. Still, having a GPU will be a significant advantage. If you have your own NVIDIA GPU, however, and wish to use that, that’s fine – you’ll need to install the drivers for your GPU, install CUDA, install cuDNN, and then install PyTorch. You could theoretically do the entire assignment with no GPUs, though this will make training GAN model much slower. 

**Installing Python 3.6+:** To use python3, make sure to install version 3.6+ on your local machine.

**Virtual environment:** If you decide to work locally, we recommend using virtual environment via [anaconda](https://www.anaconda.com/products/individual) for the project. If you choose not to use a virtual environment, it is up to you to make sure that all dependencies for the code are installed globally on your machine. To set up a conda virtual environment, run the following:

```bash    
conda create -n gan_env python=3.6 -y
conda activate gan_env
```
Install the pytorch environment following the [official instructions](https://pytorch.org/get-started/previous-versions/). Here we use PyTorch 1.7.0 and CUDA 11.0. You may also switch to other version by specifying the version number.
```bash
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
```
Install other environments in the provided `requirement.txt` file.
```bash
pip install -r requirements.txt
```

## 3. Working on the assignment

### 3.1 Basis of generative models
Before starting, learning some basics of GAN is necessary. We recommend you refer to [google document](https://developers.google.com/machine-learning/gan/generative) for a general introduction of GANs, including the overview of GAN structures, the generators and discriminators, the GAN training, and etc.
Note that some basic knowledge in [this google document](https://developers.google.com/machine-learning/gan/generative) **may appear in the final quiz.**

If you are interested, please read the related papers (e.g., [GAN](https://arxiv.org/abs/1406.2661), and [pix2pix](https://arxiv.org/abs/1611.07004)) for more details.

### 3.2 Task descriptions 
Image-to-image translation (I2I) aims to transfer images from a source domain to a target domain while preserving the content representations. I2I has drawn increasing attention and made tremendous progress in recent years because of its wide range of applications in many computer vision and image processing problems, such as image synthesis, segmentation, style transfer, restoration, and pose estimation.

In this assignment, you are going to train a generative model with the provided skeleton code. The generative model is utilized to translate the aerial photo to the map. You will be provided with a small dataset. The dataset can be found [in this link](https://drive.google.com/file/d/15fziaX7zUI1iHDBfGWflB5RVibIhcXGz/view?usp=sharing). The dataset file is named as ***maps.tar***. Please download and extract it using the command:

```
tar -xvf maps.tar
```

You will get four directories:, i.e. “train”, "val", "test" and “val_tgt”. There are 1096 images in the training dataset, 549 images in the validation dataset and 549 images in the testing dataset. The final data structure is as follows:
```
maps
├── train
|    ├── 1.jpg
|    ├── 2.jpg
|    ...
│── val
|    ├── 1.jpg
|    ├── 2.jpg
|    ...
│── test
|    ├── 1.jpg
|    ├── 2.jpg
|    ...
│── val_tgt
|    ├── 1.jpg
|    ├── 2.jpg
|    ...
```
Please move the maps directory into the main directory as in the following:
```
maps_generation
├── README.md
├── data
|    ...
├── datasets
|    ...
├── **maps**
|    ...
├── models
|    ...
├── options
|    ...
├── pytorch_ssim
|    ...
```


### 3.2 Get code
You can obtain the assignment code from the [github link]([link](https://github.com/ChongjianGE/maps_generation)).
```
git clone https://github.com/ChongjianGE/maps_generation.git
```


### 3.3 Get started
We provide the tutorial jupyter notebook `step_by_step.ipynb` to walk you through installing the environment for the  related codebase. Please follow the detailed instruction to prepare the environment.

### 3.4 Assignment works

#### 3.4.1 Baseline Implementation
**Q1: Writing the code of SSIM**
We have provided an almost-complete code for SSIM evaluation. Please fill in the specific code in the places tagged with [`Write Your Code Here`](https://github.com/ChongjianGE/maps_generation/blob/606c75db32681c56e8c1291d57c6cd59185757cc/pytorch_ssim/pytorch_ssim/__init__.py#L32). For SSIM method, we recommend you refer to the [SSIM paper](https://arxiv.org/abs/2006.13846) for implementation details. The formulation of the implementation code is as what follows:

![](https://github.com/ChongjianGE/COMP3340_Applied_DL_Note/blob/main/ssim.png?raw=true)

We have provided three groups of sample images for you to test.
```
pytorch_ssim
├── pytorch_ssim
├── sample_images
|    ├── generated_results
|    |    ├── 1.jpg
|    |    ├── 10.jpg
|    |    ├── 100.jpg
|    ├── target
|    |    ├── 1.jpg
|    |    ├── 10.jpg
|    |    ├── 100.jpg
├── test.py
```
If everything works fine, it should give you 0.502492 once you run 
```
python test.py
```

**Q2: Implementing the data-loading code for validation phase**
We have provided an almost-complete code for dataloader. Please fill in the specific code in the places tagged with [`Write Your Code Here`](https://github.com/ChongjianGE/maps_generation/blob/606c75db32681c56e8c1291d57c6cd59185757cc/data/map_dataset.py#L66).


**Q3: Implementing the training and testing code of GAN**
We have provided an almost-complete code for training and testing. Please run the code for overall training and testing.


**Q5: Evaluating the well-trained model on validation dataset**
Please write code to generate the maps on the validation dataset and evaluate the generated results with the implemented SSIM metrics.

#### 3.4.2 Baseline Improvement
**Q6: Do something extra!** ***(IMPORTANT)***
You are required to improve the baseline model with your own configuration. There are lots of ways to improve the baseline model. Here are some suggestions for you.
- **Hyper-parameter tuning.**
There are lots of important hyper-parameters.
(1) About optimizer: learning rate, batch size, warm-up and training epochs, etc.
(2) About data augmentation: flip, resize, etc.

- **Different neural network architectures for both generators and discriminators.**
You may choose a better neural network architecture (e.g., ResUNet) from the provided code, or you may also design your own customized neural networks.

- **Loss functions designs.**
You may leverage different loss terms for training (e.g., GAN loss, L1 loss, L2 loss). Besides, you can also set the weights to balance different loss terms.

- **Others.**
There are also many other possible techniques to explore. For example, you can also design your own training strategies. Besides, you can also explore some GAN training skills in this project.

**GOOD LUCK!!**

**Q8: Generate the results with your improved model**
Please train your improved model and generate the results on the test dataset (549 generated images in total).

### 3.5 Files submitting (4 items in total)
#### 1.	Prepare a final report in PDF format (no more than 4 pages)

##### a) Introduction
    Briefly introduce the task & background & related works.

##### b) Methods
    Describe what you did to improve the baseline model performance. 
    For example, this may include but not limited to:
    (i) Hyper-parameter tuning, e.g. learning rate, batch size, and training epochs.
    (ii) Different neural network architectures. 
    You may choose a better generative models (UNet, ResUNet), 
    or you may also design your own customized neural networks.
    (iv) Loss functions. 

##### c) Experiments & Analysis** (*IMPORTANT*)
    Analysis is the most important part of the report. 
    Possible analysis may include but not limited to:
    (i) Dataset analysis (dataset visualizations & statistics)
    (ii) Qualitative evaluations of your model. 
    Visualize the results of some challenging images and see if your model accurately generated the maps. 
    Failure case analysis is also suggested.
    (iii) Ablation studies. 
    Analyze why better performance can be achieved when you made some modifications, 
    e.g. hyper-parameters, model architectures, loss functions. 
    The performance (SSIM) on the validation set should be given to validate your claim.


#### 2.	Codes 
a)	All the codes. 
b)	**README.txt** that describes which python file has been added/modified.

#### 3. Models
a) Model checkpoint (.pth)
b) Model training log (loss_log.txt)

#### 4. Generated results on the test set
Please zip and submit the generated image files (on the test datasets).


```
# Please organize all these items in a zipped directory. Please rename the zipped file as your student id. For example, if your student id is 123456, then the files should be organized as follows.


123456.zip
├── report.pdf
├── maps_generation
|    ├── README.md
|    ├── data
|        ...
|    ├── datasets
|        ...
│── models_and_logs
│    │-- model.pth
│    │-- loss_log.txt
│        
├── 123456_generated_results.zip

```

### 3.6 When to submit? (TBD)

The deadline is April 3rd (Sunday).

Late submission policy: 

 10% for late assignments submitted within 1 day late.
 20% for late assignments submitted within 2 days late.
 50% for late assignments submitted within  7 days late.
100% for late assignments submitted after 7 days late.


### 3.7 Need More Support?
For any questions about the assignment, please contact the TA (Chongjian GE) via email (rhettgee@connect.hku.hk). 

## 4. Marking Scheme:

Marks will be given based on the performance that you achieve on the test set and the submitted report file. TAs will perform SSIM evaluation on the generated images. 

The marking scheme has two parts, (1) the performance ranking based on SSIM (70% marks)  and (2) the 4-page final report (30% marks): 

1. For the performance ranking part (70%): TA will rank the SSIM. The students who obtain higher SSIM will get higher marks.
   (1) Rank top-10% will get the full mark of this part.
   (2) Rank 10%-20% will get 90% mark of this part.
   (3) Rank 20%-30% will get 80% mark of this part.
   (4) Rank 30%-40% will get 70% mark of this part.
   (5) Rank 40%-50% will get 60% mark of this part.
   (6) Others will get 50% mark of this part.

2. For the 4-page final report part (30%):
   The marks will be given mainly based on the richness of the experiments & analysis.
   (1) Rich experiments + detailed analysis: 90%-100% mark of this part.
   (2) Reasonable number of experiments + analysis: 70%-80% mark of this part.
   (3) Basic analysis: 50%-60% mark of this part.
   (4) Not sufficient analysis: lower than 50%.

## Reference
1. GAN paper https://arxiv.org/abs/1406.2661
2. SSIm paper https://arxiv.org/abs/2006.13846
3. Pix2Pix GAN paper https://arxiv.org/abs/1611.07004
4. Google document on GAN introduction https://developers.google.com/machine-learning/gan/

