Optimizer Zoo: Reimplementing PyTorch Optimizers from Scratch

A **lightweight, research-oriented project** that **reimplements popular PyTorch optimizers** entirely from scratch using only **basic tensor operations.** Ideal for understanding the core mechanics of optimization algorithms and benchmarking their performance.

## 📋 Table of Contents
* [📂 Project Structure](#-project-structure)
* [🚀 Implemented Optimizers](#-implemented-optimizers)
* [🧠 Experiments](#-experiments)
* [⚙️ Usage](#-usage)
* [📈 Results](#-results)
* [🧩 Dependencies](#-dependencies)
* [📚 Future Work](#-future-work)
* [🧾 License](#-license)
* [✨ Author](#-author)

---

## 📂 Project Structure

```text
optimizer-zoo/
├── data/                  # MNIST dataset storage (ignored by git)
│   └── MNIST/raw/
├── results/               # Training outputs (ignored by git)
│   ├── checkpoints/
│   ├── logs/
│   └── loss_curves/
├── experiments/
│   ├── __init__.py
│   ├── mnist_logreg.py
│   ├── mnist_mlp.py
│   └── toy_function.py
├── models/
│   ├── logistic_regression.py
│   └── mlp.py
├── optimizers/
│   ├── __init__.py
│   ├── adabelief.py
│   ├── adagrad.py
│   ├── adam.py
│   ├── adamw.py
│   ├── amsgrad.py
│   ├── clr.py
│   ├── lookahead.py
│   ├── momentum.py
│   ├── nadam.py
│   ├── nesterov.py
│   ├── radam.py
│   ├── rmsprop.py
│   └── sgd.py
├── utils/
│   ├── __init__.py
│   ├── data_utils.py
│   ├── loss_functions.py
│   ├── plot_utils.py
│   ├── save_all_result.py
│   └── save_results.py
├── run_all_experiments.py
└── README.md
```



---

## 🚀 Implemented Optimizers

| Optimizer | Key Features |
|------------|-----------|
| **SGD** | Vanilla gradient descent |
| **Momentum** | Velocity-based acceleration |
| **Nesterov** | Nesterov Accelerated Gradient (NAG) |
| **Adagrad** | Adaptive per-parameter learning rates |
| **RMSProp** | Exponentially decayed moving average of squared gradients |
| **Adam** | Momentum + RMSProp |
| **AdamW** | Decoupled weight decay |
| **AdaBelief** | Adaptive step based on belief in the gradient |
| **AMSGrad** | Adam with non-decreasing denominator |
| **Nadam** | Adam + Nesterov momentum |
| **RAdam** | Rectified Adam for variance correction |
| **CyclicLR (CLR)** | Learning rate scheduling with cycles |
| **Lookahead** | Slow/fast optimizer combination |

---

## 🧠 Experiments

Two baseline experiments are included:

| Script | Model | Dataset |Description |
|--------|--------|-------------| ---------|
| `toy_function.py` | - | Quadratic Function | Tests convergence on f(x) = x² + 3x + 2 |
| `mnist_logreg.py` | Logistic Regression | MNIST | Linear classifier comparison |
| `mnist_mlp.py` | 2-layer MLP | MNIST | Nonlinear network comparison |

Experiment Metrics
·Training Loss: Convergence behavior over epochs

·Test Accuracy: Final performance on unseen data

·Training Time: Computational efficiency

·Convergence Speed: Iterations to reach target accuracy

---

## ⚙️ Usage

### 1. Setup Environment (Optional but Recommended)
```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Run Experiments
Run all experiments sequentially:

```bash
python run_all_experiments.py
```

Run individual experiments:
```bash
# Toy function optimization test
python -m experiments.toy_function

# Logistic regression on MNIST
python -m experiments.mnist_logreg

# MLP on MNIST  
python -m experiments.mnist_mlp
```

### 3.  Custom Usage

In each script:

```python
from optimizers import Adam
optimizer = Adam(model.parameters(), lr=1e-3)
```

---

📈 Results
Results are automatically saved to:

·results/logs/ - JSON files with detailed metrics

·results/loss_curves/ - PNG plots of training curves

Visualization Includes:

·Training loss vs. epochs

·Test accuracy vs. epochs

·Optimizer comparison plots



---

## 🧩 Dependencies
The required packages can be installed using `pip install -r requirements.txt` (recommended).

| Package | Purpose | Version |
|---------|-----------------|------|
| `torch` | Tensor operations and autograd | >=2.0.0 |
| `numpy` | Numerical computations | >=1.21.0 |
| `matplotlib` | Result visualization | >=3.5.0 |
| `tqdm` | Progress bars (optional) | 	>=4.64.0 |

---

📚 Future Work

Implement additional optimizers (Lion, Sophia, Adan)

Add support for CIFAR-10 and custom datasets

Integrate TensorBoard for real-time visualization

Add distributed training support

Create interactive comparison dashboard

Add hyperparameter optimization scripts

---

🧾 License

This project is licensed under the MIT License. Feel free to use, modify, and distribute for both academic and commercial purposes.

---

## ✨ Author

·Developed by **[Po Hung, Cheng]**
·GitHub: `[https://github.com/alexjeng0404/optimizer-zoo/blob/main/README.md]`
A comprehensive study and reimplementation of optimization algorithms for deep learning.

---