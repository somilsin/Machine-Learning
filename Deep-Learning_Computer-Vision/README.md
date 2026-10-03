# My deep learning and computer vision projects

I use this repository to understand how a neural network learns from images. I start with handwritten digits then explore face classification and adaptive sampling. I also keep a small NumPy engine where the forward passes and gradients are visible in the code.

The vision notebooks are my adaptations of MIT Introduction to Deep Learning exercises. I retain the course credit and explain the choices in my own notes. The NumPy engine implements its layers directly rather than using an automatic differentiation framework.

## What is here

1. [MNIST classification](1_MNIST_Digit_Classifications.ipynb) compares a dense baseline with a CNN. I inspect image shapes then train both models and display their predictions.
2. [Facial detection and debiasing](2_Facial_Detection_Debiasing.ipynb) compares a standard CNN with a DB VAE. I keep the distinction between mean predicted probability and classification accuracy explicit.
3. [My NumPy neural network engine](Neural-Network-Engine-From-Scratch/README.md) contains linear layers and activations with manual gradients plus losses and SGD with momentum.

## How I run it

```bash
git clone https://github.com/somilsin/Deep-Learning_Computer-Vision.git
cd Deep-Learning_Computer-Vision
python -m pip install numpy scipy matplotlib torch torchvision torchsummary tqdm h5py opencv-python tensorflow gym opik transformers datasets peft lion-pytorch ipykernel nbformat
python -m pip install "setuptools<81"
python -m pip install mitdeeplearning --no-deps --no-build-isolation
```

I select that Python environment as the notebook kernel. The notebooks print their runtime and keep metrics locally. The face notebook downloads its course training data into the local data directory and builds a contiguous array cache for fast random batches. The cache stays on disk and is read through a memory map. CUDA helps with training but the code permits CPU execution.

For the NumPy engine alone I install the existing dependency file:

```bash
python -m pip install -r Neural-Network-Engine-From-Scratch/requirements.txt
```

## My notes on the experiments

I pass raw logits into cross entropy and apply softmax only when displaying probabilities. I evaluate with gradients disabled and put models in evaluation mode so batch statistics are handled consistently.

For facial detection I treat the course demographic comparison as a limited diagnostic. I would need more held out examples and thresholded error measurements before making a broader fairness claim.

## Execution record

On 3 October 2026 I executed both notebooks on an NVIDIA RTX 3050 Ti laptop GPU with 4 GB of memory.

The MNIST run retained five dense model epochs and seven CNN epochs. The dense model achieved 96.90% test accuracy and the CNN achieved 97.38%.

The facial detection run retained two epochs for each model across the full course training dataset. The standard classifier achieved 99.76% accuracy on a sampled training batch. That is a training metric. The notebook also displays the independent group probability plots for the standard CNN and DB VAE.

The NumPy engine passed finite difference gradient checks and learned the four sample demonstration in its README. All displayed notebook outputs come from these executions.

## Course reference

The notebook exercises and helpers come from [MIT Introduction to Deep Learning](http://introtodeeplearning.com). The source credit is retained in the notebooks.
