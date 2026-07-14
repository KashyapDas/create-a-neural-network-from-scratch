# Create a Neural Network From Scratch

A foundational repository dedicated to building, training, and understanding Artificial Neural Networks (ANNs). This project strips away high-level abstractions to focus on the core mathematics and mechanics of neural network architecture, from defining layers to executing the training loop.

## 📖 Overview
This repository provides hands-on implementations of dense (fully connected) neural networks. It is designed for educational clarity, allowing developers to see exactly how data flows through a network, how loss is calculated, and how weights are updated via backpropagation.

## ✨ Features
* **Custom Architecture:** Step-by-step construction of input, hidden, and output layers.
* **Training Mechanics:** Transparent implementations of forward passes, loss calculation (e.g., CrossEntropy), and optimization algorithms (e.g., Adam).
* **Data Handling:** Examples of data preprocessing, train/test splitting, and converting standard datasets into tensor formats.
* **Model Evaluation:** Logic for interpreting raw tensor logits (using `argmax`) and mapping them back to human-readable labels.

## 🚀 Getting Started

### Prerequisites
Ensure you have the following installed in your environment:
* Python 3.x
* PyTorch
* Pandas and Scikit-Learn (for data handling and splitting)

### Installation
Clone the repository to your local machine:
```bash
git clone [https://github.com/KashyapDas/create-a-neural-network-from-scratch.git](https://github.com/KashyapDas/create-a-neural-network-from-scratch.git)
cd create-a-neural-network-from-scratch
