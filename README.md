# KAN Classifiers: Exploring the Effectiveness and Power of Kolmogorov-Arnold Networks

This project aims to explore the effectiveness and power of Kolmogorov-Arnold Networks (KANs) in classification tasks. We will conduct a series of experiments to benchmark KANs against various popular datasets and compare their performance with state-of-the-art (SOTA) architectures.

## Table of Contents
- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Using Docker](#using-docker)
- [Datasets](#datasets)
- [Models](#models)
- [Experiments](#experiments)
- [Results](#results)
- [Contributing](#contributing)

## Project Overview
This project will test the hypothesis that KANs can achieve competitive or superior performance in classification tasks compared to sota neural network architectures (e.g. Resnet, EfficientNet, ViT, etc...)

## Objectives
- Implement KANs and integrate them into a classification framework.
- Benchmark KANs on various industry-standard datasets.
- Compare the performance of KANs with SOTA architectures.
- Analyze and document the results to highlight the strengths and weaknesses of KANs.

## Repository Structure
The repository is organized as follows:

## InstallationSS
To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/chinmayembedded/KAN_for_image_classification.git
cd KAN_for_image_classification
pip install -r requirements.txt
```

## Using Docker
Follow these steps to build and run the Docker environment:

### Building the Docker Image
Navigate to the root directory of the repository and run:

```bash
docker build -f Dockerfile -t kan-img .
```

### Running the Docker Container
Once the image is built, you can run the container using:

```bash
docker run --gpus all -v <path-to-KAN-directory>:/app/kan_classifiers --name kan-env -it kan-img bash
```

This command mounts the current directory to the `/app/kan_classifiers`

## Datasets
We will be using various publicly available datasets to benchmark the models. These include but are not limited to:
- CIFAR-10
- ImageNet
- MNIST
- Custom datasets (if any)

## Models
### KAN Implementation
The KAN model is implemented in the `models` directory. Detailed documentation and configuration options can be found within the respective files.

### SOTA Models
This section is still work-in-progress!

List of SOTA models we are going to consider. Pre-trained models for Vision Transformers (ViT), ResNet, and others are available in the `models/sota` directory. These models can be fine-tuned or evaluated directly on the provided datasets.

## Experiments


### Running Experiments
To run an experiment, use the following command:

```bash
python experiments/scripts/run_experiment.py --config=configs/experiment1.yaml
```

Experiment configurations are stored in YAML files within the `experiments/scripts/configs` directory.

## Results
Results of the experiments, including performance metrics and model checkpoints, will be stored in the `experiments/results` directory. Detailed analysis and visualizations can be found in the `notebooks` directory.