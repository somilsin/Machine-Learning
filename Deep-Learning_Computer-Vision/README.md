<div align="center">

# 👁️ Deep Learning and Computer Vision

### Image classification, adaptive sampling and visible gradients

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
[![X](https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/Skywalkerlyzv)
[![Medium](https://img.shields.io/badge/Medium-000000?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@thesomilsinghofficial)
[![Substack](https://img.shields.io/badge/Substack-FF6719?style=for-the-badge&logo=substack&logoColor=white)](https://thesomilsingh.substack.com/)

[Open this project](https://github.com/somilsin/Machine-Learning/tree/main/Deep-Learning_Computer-Vision) · [My GitHub](https://github.com/somilsin) · [Main Portfolio](https://somilsin.github.io/Artificial-Intelligence/portfolio/)

</div>

---

## 📖 About This Repository

I use these notebooks to work through how neural networks learn from images. I started with handwritten digits, then explored face classification, adaptive sampling and generative models.

The vision notebooks are my adaptations of MIT Introduction to Deep Learning exercises. The NumPy engine is where I write out the layers and gradients directly.

---

## 🚀 Key Implementations

* Dense and convolutional digit classifiers
* Face classification with adaptive sampling
* A NumPy neural network engine with manual gradients
* A variational autoencoder that generates new digits from a 2D latent space

---

## 🎓 Project Guide

1. [MNIST classification](1_MNIST_Digit_Classifications.ipynb) compares a dense baseline with a CNN. I inspect image shapes then train both models and display their predictions.
2. [Facial detection and debiasing](2_Facial_Detection_Debiasing.ipynb) compares a standard CNN with a DB VAE. I keep the distinction between mean predicted probability and classification accuracy explicit.
3. [My NumPy neural network engine](Neural-Network-Engine-From-Scratch/README.md) contains linear layers and activations with manual gradients plus losses and SGD with momentum.
4. [My variational autoencoder from scratch](Variational-Autoencoder-From-Scratch/README.md) is the companion code for my VAE lecture on YouTube. I train it on binarized digits, then sample new digits from the prior and decode the whole 2D latent space.

---

## 🛠️ Tech Stack

<p align="center">
<img src="https://skillicons.dev/icons?i=python,pytorch,tensorflow&theme=dark" alt="Python, PyTorch, TensorFlow" />
</p>

`Python` · `PyTorch` · `TensorFlow`

---

## ⚙️ Getting Started

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

For the variational autoencoder I use its own dependency file and script:

```bash
python -m pip install -r Variational-Autoencoder-From-Scratch/requirements.txt
python Variational-Autoencoder-From-Scratch/vae.py --epochs 10 --latent-dim 2
```

---

## 📊 Results and Notes

### Training choices

I pass raw logits into cross entropy and apply softmax only when displaying probabilities. I evaluate with gradients disabled and put models in evaluation mode so batch statistics are handled consistently.

For facial detection I treat the course demographic comparison as a limited diagnostic. I would need more held out examples and thresholded error measurements before making a broader fairness claim.

### My runs

On 3 October 2026 I executed both notebooks on an NVIDIA RTX 3050 Ti laptop GPU with 4 GB of memory.

The MNIST run retained five dense model epochs and seven CNN epochs. The dense model achieved 96.90% test accuracy and the CNN achieved 97.38%.

The facial detection run retained two epochs for each model across the full course training dataset. The standard classifier achieved 99.76% accuracy on a sampled training batch. That is a training metric. The notebook also displays the independent group probability plots for the standard CNN and DB VAE.

The NumPy engine passed finite difference gradient checks and learned the four sample demonstration in its README. All displayed notebook outputs come from these executions.

On 8 October 2026 I trained the variational autoencoder for 10 epochs on my laptop CPU in 2 minutes 16 seconds. It reached a test loss of 152.60 nats per image, made of 146.70 reconstruction and 5.90 KL. Its folder keeps the loss history and the generated figures.

---

## 📚 References and Credit

The notebook exercises and helpers come from [MIT Introduction to Deep Learning](http://introtodeeplearning.com). The source credit is retained in the notebooks.

The variational autoencoder follows Kingma and Welling's [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114). Its references are listed in its own README.

---

<div align="center">

### Get In Touch

If you'd like to chat about the work, you can reach me on [LinkedIn](https://linkedin.com/in/somil-singh).

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/somil-singh)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:somils@andrew.cmu.edu)
[![X](https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/Skywalkerlyzv)
[![Medium](https://img.shields.io/badge/Medium-000000?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@thesomilsinghofficial)
[![Substack](https://img.shields.io/badge/Substack-FF6719?style=for-the-badge&logo=substack&logoColor=white)](https://thesomilsingh.substack.com/)

*Thanks for stopping by!*

</div>
