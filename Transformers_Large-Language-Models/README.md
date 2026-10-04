<!-- Shared decorative layout inspired by my original vision README and profile README. -->
<div align="center">

<h1>🤖 Transformers and Large Language Models</h1>
<h3><code>Sequence prediction, music generation and language adaptation</code></h3>

<img src="https://readme-typing-svg.demolab.com/?font=Fira+Code&weight=500&size=22&pause=1000&color=58A6FF&center=true&vCenter=true&width=900&lines=Sequence%20prediction%2C%20music%20generation%20and%20language%20adaptation;Learn+it.+Build+it.+Explain+it." alt="Sequence prediction, music generation and language adaptation" />

<p>
<img src="https://img.shields.io/badge/Machine%20Learning-6E40C9?style=for-the-badge" alt="Machine Learning" />
<img src="https://img.shields.io/badge/Maintained%20by%20Somil%20Singh-58A6FF?style=for-the-badge&logo=github&logoColor=white" alt="Maintained by Somil Singh" />

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge" alt="Python" />
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge" alt="PyTorch" />
</p>

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/somil-singh)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:somils@andrew.cmu.edu)

[Open this project](https://github.com/somilsin/Machine-Learning/tree/main/Transformers_Large-Language-Models) · [My GitHub](https://github.com/somilsin) · [My portfolio](https://somilsin.github.io/Artificial-Intelligence/portfolio/)

</div>

<br>

## 📖 About This Repository

---

I study sequence prediction through music notation generation and language model adaptation. I use these notebooks to understand the training choices and inspect the generated outputs.

<br>

## 🚀 Key Implementations

---

* Music generation with a recurrent sequence model
* Language model adaptation with low rank adapters
* Saved text and audio from clearly labelled local runs

<br>

## 🎓 Project Guide

---

### Overview

I use this repository to study sequence prediction and language model adaptation. The first notebook learns ABC music notation with an LSTM. The second adapts a pretrained language model with LoRA and includes an optional style judge.

These notebooks are my working adaptations of MIT Introduction to Deep Learning labs. I keep the course copyright and data helper credit. I explain the pipeline in my own notes and distinguish model training from the use of pretrained weights.

### What is here

1. [Music generation](1_Music_Generation.ipynb) covers character encoding and shifted sequence targets then LSTM training and generation. Generated ABC text is displayed directly.
2. [Language model adaptation](2_LLM_Finetuning.ipynb) covers chat formatting and tokenization then LoRA training and held out evaluation using LiquidAI LFM2 with 1.2 billion parameters.

<br>

## 🛠️ Tech Stack

---

<p align="center">
<img src="https://skillicons.dev/icons?i=python,pytorch&theme=dark" alt="Python, PyTorch" />
</p>

`Python` · `PyTorch`

<br>

## ⚙️ Getting Started

---

### How I run it

```bash
git clone https://github.com/somilsin/Machine-Learning.git
cd Machine-Learning/Transformers_Large-Language-Models
python -m pip install numpy scipy matplotlib music21 torch transformers datasets peft accelerate lion-pytorch tqdm opik seaborn pandas opencv-python tensorflow gym ipykernel nbformat
python -m pip install "setuptools<81"
python -m pip install mitdeeplearning --no-deps --no-build-isolation
```

I use that environment as the notebook kernel. The music notebook defaults to 3,000 updates with 1,024 hidden units. MUSIC_STEPS and MUSIC_HIDDEN_SIZE can be set before starting the kernel for a smaller run. The language notebook defaults to 200 updates per style and accepts LLM_STEPS as an override. LLM_MODEL_ID selects the pretrained model and LLM_CONTEXT sets the training context length. The saved outputs record the selected settings.

Audio playback uses abc2midi and timidity when available. Otherwise I parse the notation with music21 and display previews of up to eight seconds using simple sine tones at 120 quarter notes per minute. Invalid notation is reported explicitly.

The judge requires OPENROUTER_API_KEY and tracing requires OPIK_API_KEY. I read credentials from environment variables. Those services are optional and the notebook does not invent scores when they are unavailable.

<br>

## 📝 My Notes and Results

---

### My notes on interpretation

The LSTM loss measures prediction of the next character rather than how pleasant a tune sounds. I inspect complete generated songs and would test on held out songs before comparing settings.

LoRA updates a small set of adapter parameters while the base model stays frozen. Training the same adapter on a second style is a sequential adaptation experiment. I would use fresh adapters for an independent comparison.

### Execution record

On 3 October 2026 I executed both notebooks on an NVIDIA RTX 3050 Ti laptop GPU with 4 GB of memory.

For music I used a smaller local run with 600 updates and 256 hidden units. The recorded final batch loss was 1.1722. The notebook displays generated ABC text and audio previews parsed from that text.

For language adaptation I used LiquidAI LFM2 350M with a context length of 256 tokens and 20 adapter updates for each style. There were 245,760 trainable parameters out of 354,729,728 total parameters. The final course Yoda text cross entropy loss was 3.97. The cell retains the course label loglikelihood but its calculation is cross entropy so I interpret it as a loss.

These smaller runs demonstrate execution rather than convergence at the full settings. Generated responses are saved in the notebook. External judge scoring and tracing were skipped because the service keys were not configured. No judge scores were invented.

<br>

## 📚 References and Credit

---

### Reference and license

The exercises and helpers come from [MIT Introduction to Deep Learning](http://introtodeeplearning.com). I retain the original copyright notices. The repository license remains in [LICENSE](LICENSE).

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
