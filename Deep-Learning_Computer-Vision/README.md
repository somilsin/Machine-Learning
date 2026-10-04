<!-- Shared decorative layout inspired by my original vision README and profile README. -->
<div align="center">

<h1>👁️ Deep Learning and Computer Vision</h1>
<h3><code>Image classification, adaptive sampling and visible gradients</code></h3>

<img src="https://readme-typing-svg.demolab.com/?font=Fira+Code&weight=500&size=22&pause=1000&color=58A6FF&center=true&vCenter=true&width=900&lines=Image%20classification%2C%20adaptive%20sampling%20and%20visible%20gradients;Learn+it.+Build+it.+Explain+it." alt="Image classification, adaptive sampling and visible gradients" />

<p>
<img src="https://img.shields.io/badge/Machine%20Learning-6E40C9?style=for-the-badge" alt="Machine Learning" />
<img src="https://img.shields.io/badge/Maintained%20by%20Somil%20Singh-58A6FF?style=for-the-badge&logo=github&logoColor=white" alt="Maintained by Somil Singh" />

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge" alt="Python" />
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge" alt="PyTorch" />
<img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge" alt="TensorFlow" />
</p>

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/somil-singh)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:somils@andrew.cmu.edu)

[Open this project](https://github.com/somilsin/Machine-Learning/tree/main/Deep-Learning_Computer-Vision) · [My GitHub](https://github.com/somilsin) · [My portfolio](https://somilsin.github.io/Artificial-Intelligence/portfolio/)

</div>

<br>

## 📖 About This Repository

---

I study how neural networks learn from images through digit classification and face classification. I also maintain a NumPy engine where the layers and gradients are visible.

<br>

## 🚀 Key Implementations

---

* Dense and convolutional digit classifiers
* Face classification with adaptive sampling
* A NumPy neural network engine with manual gradients

<br>

## 🎓 Project Guide

---

### Overview

I use this repository to understand how a neural network learns from images. I start with handwritten digits then explore face classification and adaptive sampling. I also keep a small NumPy engine where the forward passes and gradients are visible in the code.

The vision notebooks are my adaptations of MIT Introduction to Deep Learning exercises. I retain the course credit and explain the choices in my own notes. The NumPy engine implements its layers directly rather than using an automatic differentiation framework.

### What is here

1. [MNIST classification](1_MNIST_Digit_Classifications.ipynb) compares a dense baseline with a CNN. I inspect image shapes then train both models and display their predictions.
2. [Facial detection and debiasing](2_Facial_Detection_Debiasing.ipynb) compares a standard CNN with a DB VAE. I keep the distinction between mean predicted probability and classification accuracy explicit.
3. [My NumPy neural network engine](Neural-Network-Engine-From-Scratch/README.md) contains linear layers and activations with manual gradients plus losses and SGD with momentum.

<br>

## 🛠️ Tech Stack

---

<p align="center">
<img src="https://skillicons.dev/icons?i=python,pytorch,tensorflow&theme=dark" alt="Python, PyTorch, TensorFlow" />
</p>

`Python` · `PyTorch` · `TensorFlow`

<br>

## ⚙️ Getting Started

---

### How I run it

```bash
git clone https://github.com/somilsin/Machine-Learning.git
cd Machine-Learning/Deep-Learning_Computer-Vision
python -m pip install numpy scipy matplotlib torch torchvision torchsummary tqdm h5py opencv-python tensorflow gym opik transformers datasets peft lion-pytorch ipykernel nbformat
python -m pip install "setuptools<81"
python -m pip install mitdeeplearning --no-deps --no-build-isolation
```

I select that Python environment as the notebook kernel. The notebooks print their runtime and keep metrics locally. The face notebook downloads its course training data into the local data directory and builds a contiguous array cache for fast random batches. The cache stays on disk and is read through a memory map. CUDA helps with training but the code permits CPU execution.

For the NumPy engine alone I install the existing dependency file:

```bash
python -m pip install -r Neural-Network-Engine-From-Scratch/requirements.txt
```

<br>

## 📝 My Notes and Results

---

### My notes on the experiments

I pass raw logits into cross entropy and apply softmax only when displaying probabilities. I evaluate with gradients disabled and put models in evaluation mode so batch statistics are handled consistently.

For facial detection I treat the course demographic comparison as a limited diagnostic. I would need more held out examples and thresholded error measurements before making a broader fairness claim.

### Execution record

On 3 October 2026 I executed both notebooks on an NVIDIA RTX 3050 Ti laptop GPU with 4 GB of memory.

The MNIST run retained five dense model epochs and seven CNN epochs. The dense model achieved 96.90% test accuracy and the CNN achieved 97.38%.

The facial detection run retained two epochs for each model across the full course training dataset. The standard classifier achieved 99.76% accuracy on a sampled training batch. That is a training metric. The notebook also displays the independent group probability plots for the standard CNN and DB VAE.

The NumPy engine passed finite difference gradient checks and learned the four sample demonstration in its README. All displayed notebook outputs come from these executions.

<br>

## 📚 References and Credit

---

### Course reference

The notebook exercises and helpers come from [MIT Introduction to Deep Learning](http://introtodeeplearning.com). The source credit is retained in the notebooks.

<br>

## 🗂️ Explore My Other Work

---

| [Artificial Intelligence](https://github.com/somilsin/Artificial-Intelligence) | [Machine Learning](https://github.com/somilsin/Machine-Learning) | [Computer Vision](https://github.com/somilsin/Computer-Vision) | [Learning Archive](https://github.com/somilsin/Learning-Archive) |
| :---: | :---: | :---: | :---: |

<br>

<div align="center">

### Get In Touch

I share my learning and projects here. Connect with me on [LinkedIn](https://linkedin.com/in/somil-singh) or explore [my portfolio](https://somilsin.github.io/Artificial-Intelligence/portfolio/).

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/somil-singh)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:somils@andrew.cmu.edu)

*Thanks for stopping by!*

</div>
