# MNIST Handwritten Digit Classification using PyTorch

## Project Overview

This project demonstrates how to build a simple neural network using **PyTorch** to classify handwritten digits from the **MNIST dataset**.

The project covers the complete deep learning workflow:

- Loading the dataset
- Creating DataLoaders
- Building a neural network
- Training the model
- Testing model accuracy
- Saving the trained model

---

## Libraries Used

- PyTorch
- TorchVision
- Matplotlib

---

## Neural Network Architecture

Architecture:

```
Input (28×28)

↓

Flatten

↓

Linear (784 → 128)

↓

ReLU

↓

Linear (128 → 64)

↓

ReLU

↓

Linear (64 → 10)
```

---

## Project Structure

```
MNIST/
│
├── data/
│
├── train.ipynb
│
├── mnist_model.pth
│
└── README.md
```
---

## Output

- Successfully trains a neural network on the MNIST dataset.
- Achieves approximately **97.3% test accuracy**.
- Saves the trained model for future inference.


---