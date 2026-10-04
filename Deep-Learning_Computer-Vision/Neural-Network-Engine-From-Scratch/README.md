<!-- Shared decorative layout inspired by my original vision README and profile README. -->
<div align="center">

<h1>🧠 NumPy Neural Network Engine</h1>
<h3><code>Forward passes, gradients and optimization from first principles</code></h3>

<img src="https://readme-typing-svg.demolab.com/?font=Fira+Code&weight=500&size=22&pause=1000&color=58A6FF&center=true&vCenter=true&width=900&lines=Forward%20passes%2C%20gradients%20and%20optimization%20from%20first%20principles;Learn+it.+Build+it.+Explain+it." alt="Forward passes, gradients and optimization from first principles" />

<p>
<img src="https://img.shields.io/badge/Machine%20Learning-6E40C9?style=for-the-badge" alt="Machine Learning" />
<img src="https://img.shields.io/badge/Maintained%20by%20Somil%20Singh-58A6FF?style=for-the-badge&logo=github&logoColor=white" alt="Maintained by Somil Singh" />

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge" alt="Python" />
</p>

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/somil-singh)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:somils@andrew.cmu.edu)
[![X](https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/Skywalkerlyzv)
[![Medium](https://img.shields.io/badge/Medium-000000?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@thesomilsinghofficial)
[![Substack](https://img.shields.io/badge/Substack-FF6719?style=for-the-badge&logo=substack&logoColor=white)](https://thesomilsingh.substack.com/)

[Open this project](https://github.com/somilsin/Machine-Learning/tree/main/Deep-Learning_Computer-Vision/Neural-Network-Engine-From-Scratch) · [My GitHub](https://github.com/somilsin) · [My portfolio](https://somilsin.github.io/Artificial-Intelligence/portfolio/)

</div>

<br>

## 📖 About This Repository

---

I implement a small neural network engine with NumPy and SciPy. The layers calculate their forward and backward passes directly so I can inspect every gradient.

<br>

## 🚀 Key Implementations

---

* Linear layers and activations with manual gradients
* Loss functions and stochastic gradient descent with momentum
* Batch normalization and multilayer perceptron examples

<br>

## 🎓 Project Guide

---

### Overview

I built this small engine to make the calculations inside a neural network visible. Each layer stores the values needed by its backward pass. I can follow the gradient from the loss through the model and into the SGD update.

I use NumPy for array operations and SciPy for the error function and stable sigmoid. The engine does not use an automatic differentiation framework.

### What I implemented

1. Linear layers compute a batch matrix product and its input gradient plus weight and bias gradients.
2. Identity and sigmoid then tanh and ReLU provide simple activations. GELU and Swish add smooth alternatives. Swish also exposes its gate gradient.
3. Softmax subtracts each row maximum before exponentiation. Its backward pass uses a Jacobian vector product so I do not need to materialise every class interaction.
4. MSE averages over all output elements. Cross entropy uses log sum exp to remain finite for extreme logits and averages over examples.
5. BatchNorm1d uses batch statistics in training and running statistics in inference. Its backward pass follows the selected mode.
6. SGD selects layers that have weights and biases then updates them with optional momentum.
7. MLP0 and MLP1 then MLP4 show models with different depths. I initialise weights randomly so the ReLU networks can start learning.

### My gradient checks

I compared the analytical gradients with central finite differences for every activation and both losses plus the linear layer and BatchNorm1d. The largest absolute difference was below 1.1e-10. I also checked weight updates for all three model depths and the second momentum update.

For logits of positive and negative 1,000 the cross entropy remained finite at 2,000. That check matters because softmax probabilities can underflow even when the expected loss is finite.

### What I would extend next

The current SGD implementation updates linear layers. It does not automatically update BatchNorm scale and shift or the Swish gate. Those gradients are available for inspection but a model that learns those parameters needs an optimizer extension.

The public MLP examples retain ReLU at the output for compatibility with their original design. For a general classification model I would normally use an unrestricted final linear layer for logits.

<br>

## 🛠️ Tech Stack

---

<p align="center">
<img src="https://skillicons.dev/icons?i=python&theme=dark" alt="Python" />
</p>

`Python`

<br>

## ⚙️ Getting Started

---

### A complete example I can run

From this directory I install the dependencies and run the example below:

```bash
python -m pip install -r requirements.txt
```

```python
import numpy as np
from nnkit.models import MLP1
from nnkit.nn import CrossEntropyLoss
from nnkit.optim import SGD

np.random.seed(7)
X = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
Y = np.eye(2)[[0, 0, 1, 1]]
model = MLP1()
criterion = CrossEntropyLoss()
optimizer = SGD(model, lr=0.08, momentum=0.9)
initial_loss = criterion.forward(model.forward(X), Y)

for step in range(300):
    loss = criterion.forward(model.forward(X), Y)
    model.backward(criterion.backward())
    optimizer.step()

final_loss = criterion.forward(model.forward(X), Y)
predicted = np.argmax(model.forward(X), axis=1)
accuracy = np.mean(predicted == np.argmax(Y, axis=1))
print(f"Loss: {initial_loss:.6f} to {final_loss:.6f}")
print(f"Accuracy: {accuracy:.0%}")
```

Output from the local execution on 3 October 2026:

```text
Loss: 0.468641 to 0.000260
Accuracy: 100%
```

This is a tiny training example with evaluation on the same four samples. It checks that the engine learns rather than measuring generalisation.

<br>

## 📝 My Notes and Results

---

The finite difference gradient check reached a maximum error of approximately 1.1 × 10⁻¹⁰. The four sample demonstration reached 100% accuracy with its loss decreasing from 0.468641 to 0.000260.

<br>

## 📚 References and Credit

---

I retain the existing source context. This implementation calculates gradients directly without an automatic differentiation framework.

<br>

<div align="center">

### Get In Touch

I share my learning and projects here. Connect with me on [LinkedIn](https://linkedin.com/in/somil-singh) or explore [my portfolio](https://somilsin.github.io/Artificial-Intelligence/portfolio/).

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/somil-singh)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:somils@andrew.cmu.edu)
[![X](https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/Skywalkerlyzv)
[![Medium](https://img.shields.io/badge/Medium-000000?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@thesomilsinghofficial)
[![Substack](https://img.shields.io/badge/Substack-FF6719?style=for-the-badge&logo=substack&logoColor=white)](https://thesomilsingh.substack.com/)

*Thanks for stopping by!*

</div>
