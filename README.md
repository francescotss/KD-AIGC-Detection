# Knowledge Distillation for AI-Generated Image Detection: A Comparative Analysis of Continual Learning Strategies

![pipeline](https://github.com/francescotss/KD-AIGC-Detection/assets/44005266/76e76c6b-4635-4ab2-8e23-26478a4b689e)

In this work, we implemented a robust image classification system for detecting AI-generated images using Knowledge Distillation. It facilitates incremental training in a continual learning scenario while avoiding catastrophic forgetting.

#### Abstract
> Over the past few years, image-generative models have made tremendous strides. Today, these models are ubiquitous in our society, thanks to their widespread adoption across various fields, flooding the web, social media, and newspapers with potentially harmful content. To combat this threat, numerous high-quality AI-generated image detectors are being proposed. However, most of the models can only detect images synthesized by generative architectures available during training. Attempting to simply add a new architecture to the model's knowledge is a challenge due to the catastrophic forgetting problem, a phenomenon in which a neural network partially or completely loses its previous knowledge while trying to learn a new task. This forces us to train a new detection model from scratch whenever a new generative model is presented,  putting high pressure on the image forensics community to continually develop new detection models to keep up with the fast-paced development of new generative networks. In this work, we present an alternative approach using Continual Learning. This growing field of research aims to develop adaptive algorithms capable of learning from a stream of data without incurring into the catastrophic forgetting problem. Our objective is to investigate the feasibility of an AI-generated image detection model that can extend its capabilities to multiple generative sources. We test a state-of-the-art knowledge distillation-based method over a stream of 13 datasets and compare it with different alternative strategies. Our extensive experimental results demonstrate that the proposed architecture effectively mitigates catastrophic forgetting, outperforming the other state-of-the-art methods.

## Requirements and Usage

The code is designed to be executed on Google Colab, just open [colab_notebook.ipynb](notebooks/colab_notebook.ipynb). All the instructions to run trainings and evaluations are provided within the notebook.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/francescotss/KD-AIGC-Detection/blob/main/notebooks/colab_notebook.ipynb)


### Notebook structure

1. **Startup**: Start colab and install dependencies
2. -6. **Dataset, TL, KD, EWC, Evaluate**:  Dataset, methods and evaluation function definition
7. **Workspace**: In this section, it is possible to configure and run trainings and evaluations

### Expected dataset structure
Each dataset must be stored in
`datasets/<type>/<dataset_name>.zip`. They will be extracted at runtime in `$destination_path/<dataset_name>`


```
train/
---0_real/
------img1.png
------img2.png
---1_fake/
------img1.png
------img2.png
val/
---0_real/
------img1.png
------img2.png
---1_fake/
------img1.png
------img2.png
test/
---0_real/
------img1.png
------img2.png
---1_fake/
------img1.png
------img2.png
```
