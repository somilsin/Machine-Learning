<div align="center">

# 🤖 Transformers and Large Language Models

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge) ![Sequence Models](https://img.shields.io/badge/Sequence%20Models-6E40C9?style=for-the-badge)

**By [Somil Singh](https://github.com/somilsin)**

</div>

[← Machine Learning](../README.md)

I use this repository to study sequence prediction and language model adaptation. The first notebook learns ABC music notation with an LSTM. The second adapts a pretrained language model with LoRA and includes an optional style judge.

These notebooks are my working adaptations of MIT Introduction to Deep Learning labs. I keep the course copyright and data helper credit. I explain the pipeline in my own notes and distinguish model training from the use of pretrained weights.

## What is here

1. [Music generation](1_Music_Generation.ipynb) covers character encoding and shifted sequence targets then LSTM training and generation. Generated ABC text is displayed directly.
2. [Language model adaptation](2_LLM_Finetuning.ipynb) covers chat formatting and tokenization then LoRA training and held out evaluation using LiquidAI LFM2 with 1.2 billion parameters.

## How I run it

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

## My notes on interpretation

The LSTM loss measures prediction of the next character rather than how pleasant a tune sounds. I inspect complete generated songs and would test on held out songs before comparing settings.

LoRA updates a small set of adapter parameters while the base model stays frozen. Training the same adapter on a second style is a sequential adaptation experiment. I would use fresh adapters for an independent comparison.

## Execution record

On 3 October 2026 I executed both notebooks on an NVIDIA RTX 3050 Ti laptop GPU with 4 GB of memory.

For music I used a smaller local run with 600 updates and 256 hidden units. The recorded final batch loss was 1.1722. The notebook displays generated ABC text and audio previews parsed from that text.

For language adaptation I used LiquidAI LFM2 350M with a context length of 256 tokens and 20 adapter updates for each style. There were 245,760 trainable parameters out of 354,729,728 total parameters. The final course Yoda text cross entropy loss was 3.97. The cell retains the course label loglikelihood but its calculation is cross entropy so I interpret it as a loss.

These smaller runs demonstrate execution rather than convergence at the full settings. Generated responses are saved in the notebook. External judge scoring and tracing were skipped because the service keys were not configured. No judge scores were invented.

## Reference and license

The exercises and helpers come from [MIT Introduction to Deep Learning](http://introtodeeplearning.com). I retain the original copyright notices. The repository license remains in [LICENSE](LICENSE).
